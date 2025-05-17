import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from ETrain.utils.LLaVA.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from ETrain.utils.LLaVA.conversation import conv_templates, SeparatorStyle
from ETrain.Models.LLaVA.builder import load_pretrained_model
from ETrain.utils.LLaVA.utils import disable_torch_init
from ETrain.utils.LLaVA.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import copy
import random
import pandas


# Ability LoRA selection prompt
ROUTING_PROMPT = """You are a helpful assistant router. There are four expert models, each specializing in one of the following domains: OCR, math & logic, counting, and GUI navigation.
Your task is to select the most suitable model based on the provided visual content, user question, and model descriptions. Consider the expertise of each model carefully, and select the one best equipped to handle the given question. 

**Important Instructions:**  
- Respond **only** with the letter (A,B,C,D) corresponding to the most suitable model.  
- Do **not** attempt to answer the user's question directly.  

**Model Pool:**  
- **A**: This model excels in OCR tasks, including text extraction, handwriting recognition, and document analysis.
- **B**: This model excels in counting the number of objects in the image. However, it struggles to exact text in an image.
- **C**: This model is an expert in math and logic, including solving equations, geometry, and logical reasoning. It is capable of on puzzle test figures, algebraic reasoning over functional plots, and scientific reasoning with academic paper figures.
- **D**: This model is an expert in GUI navigation, including identifying buttons, text fields, and other UI elements from screen shots. It is capable of giving coordinates of the elements in the image and conduct action on the elements.

Here is the user's question: """

PROMPT_AFTER_QUESTION = "You only need to select the suitable model and do not answer the question. JUST answer with the model's letter from the given choices directly."

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, args, questions, image_folder, tokenizer, image_processor, model_config):
        self.qf = args.qf
        self.questions = questions
        self.image_folder = image_folder
        # print("args.result_folders:", args.result_folders)
        self.result_folders = args.result_folders
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.names = "OCR Count Math APP".split(' ')
        self.par = [f"{name}_{self.qf}" for name in self.names]

        self.ad = []
        for rdir in os.listdir(self.result_folders):
            if rdir not in self.par:
                continue
            self.ad.append(os.path.join(self.result_folders, rdir))
        # change the order of self.ad to be the same as self.par
        self.ad = sorted(self.ad, key=lambda x: self.par.index(os.path.basename(x)))
        
        self.ans = [] # a list of dicts, each dict is a model's answers for all questions, len(self.ans) == number of models
        for i in range(len(self.ad)):
            self.ans.append([json.loads(l) for l in open(os.path.join(self.ad[i], 'Finetune', 'merge.jsonl'), 'r').readlines()])
        for i in range(len(self.ans)):
            self.ans[i] = {x['question_id']: x for x in self.ans[i]}

        qs = [] # questions with answers, each question is a dict with keys: question_id, text, image(optional), 
        # ans (a list of answers for all models), answer (GT answer)
        for i in range(len(self.questions)):
            q = self.questions[i] # a dict with keys: question_id, text, image, ans
            qid = q['question_id'] # question_id
            flag = False
            for j in range(len(self.ans)): # for each model's answers
                if qid not in self.ans[j]: 
                    print(f"Question {qid} not found in {self.names[j]} answers.")
                    # print(self.ans[j][qid])
                    # exit(0)
                    flag = True
                    break
            if flag:
                continue
                
            q['ans'] = [f"{self.ans[j][qid]['text']}" for j in range(len(self.ans))]
            qs.append(q)
        self.questions = qs
        # print(qs)

    def __getitem__(self, index):
        line = self.questions[index]
        qs_text = line["text"] 
        ans = line['ans']
        idx = line["question_id"]
        qs_text = qs_text.replace("<image>", "").strip() # for ScienceQA, remove <image> token
        routing_qs = copy.deepcopy(ROUTING_PROMPT) + qs_text + '\n' + PROMPT_AFTER_QUESTION
        
        if not line.get('image'):
            image_tensor = "None"
        else:
            image_file = line["image"]
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]
            if self.model_config.mm_use_im_start_end:
                qs_text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_text
                routing_qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + routing_qs
            else:
                qs_text = DEFAULT_IMAGE_TOKEN + '\n' + qs_text
                routing_qs = DEFAULT_IMAGE_TOKEN + '\n' + routing_qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        conv_router = conv_templates[args.conv_mode].copy()
        conv_router.append_message(conv_router.roles[0], routing_qs)
        conv_router.append_message(conv_router.roles[1], None)
        routing_qs = conv_router.get_prompt()
        

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        routing_input_ids = tokenizer_image_token(routing_qs, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        return input_ids, routing_input_ids, image_tensor

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(args, questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(args, questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def choose_ans(routing_outputs, ans_candidates):
    # check the rounting outputs is legal
    if sum(1 for c in routing_outputs if c not in 'ABCD') > 1:
        print(f'[Warning] Routing outputs {routing_outputs} are not legal')
       
        return ans_candidates[random.randint(0, 4)]
        # assert False
    if 'A' in routing_outputs:
        return ans_candidates[0]
    elif 'B' in routing_outputs:
        return ans_candidates[1]
    elif 'C' in routing_outputs:
        return ans_candidates[2]
    elif 'D' in routing_outputs:
        return ans_candidates[3]
    # elif 'E' in routing_outputs:
    #     return ans_candidates[4]
    else:
        print(f'[Warning] Routing outputs {routing_outputs} are not legal')
        return ans_candidates[random.randint(0, 3)]
        # assert False

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode: # not use
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(args, questions, args.image_folder, tokenizer, image_processor, model.config, num_workers=args.num_worker)
    # Initialize a dictionary to count agent_selection occurrences
    agent_selection_count = {}
    
    excel_ori = pandas.read_excel('llava_v1.5_7b_MathVista_MINI.xlsx') #  a DataFrame
    excel = copy.deepcopy(excel_ori)
    for (input_ids, routing_input_ids, image_tensor), line in tqdm(zip(data_loader, questions), total=len(questions)):
        # print(type(image_tensor))
        if isinstance(image_tensor, str|tuple):
            # continue
            image_tensor = None
        idx = line["question_id"]
        cur_prompt = line["text"]
        prompts = [[cur_prompt.replace("<image>\n", "").lower()]]
        ans_candidates = line['ans']
        
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        routing_input_ids = routing_input_ids.to(device='cuda', non_blocking=True)
        GT_ans = line['answer']
        
        with torch.inference_mode():
            output_ids = model.generate(
                routing_input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True) if image_tensor is not None else None,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True)
        
        routing_token_len = routing_input_ids.shape[1]
        n_diff_routing_input_output = (routing_input_ids != output_ids[:, :routing_token_len]).sum().item()
        if n_diff_routing_input_output > 0:
            print(f'[Warning] {n_diff_routing_input_output} routing_input_ids are not the same as the routing_input_ids')

        routing_outputs = tokenizer.batch_decode(output_ids[:, routing_token_len:], skip_special_tokens=True)[0]
        routing_outputs = routing_outputs.strip()
        print(f"Routing outputs: {routing_outputs}")
        # save all routing outputs
        final_ans = choose_ans(routing_outputs, ans_candidates)
        # if args.qf=='count' or args.qf=='ocr' or args.qf == 'app':
        # update agent_selection_count
        if routing_outputs in agent_selection_count:
            agent_selection_count[routing_outputs] += 1
        else:
            agent_selection_count[routing_outputs] = 1
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                "prompt": cur_prompt,
                                "agent_selection": routing_outputs,
                                "text": final_ans,
                                "answer_id": ans_id,
                                "model_id": model_name,
                                "metadata": {}}) + "\n")
        if args.qf == 'math':
            id_excel = int(idx.replace("testmini_",""))
            excel.at[id_excel, 'prediction'] = final_ans

            
    ans_file.close()
    if args.qf=='math':
        excel.to_excel(args.output_xlsx, index=False)
        print('Excel saved!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folders', type=str, default='results/CoIN/LLaVA')
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="checkpoints/LLaVA/Vicuna/vicuna-7b-v1.5")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--qf", type=str, default="sqa")
    parser.add_argument("--num_worker", type=int, default=4)
    parser.add_argument("--output_xlsx", type=str, default="llava_v1.5_7b_MathVista_MINI.xlsx")
    args = parser.parse_args()

    eval_model(args)
