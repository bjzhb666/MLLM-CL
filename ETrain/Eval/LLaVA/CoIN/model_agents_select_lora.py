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

selection_prompt = "You are a helpful assistant router. Based on the visual content, questions, and model pool the user provides, especially the question, you need to consider the expertise of these models to select the most suitable model to help you answer the questions. JUST answer with the model's letter from the given choices directly. Note that your job is to select the most suitable model rather than asking the question directly.\nModel pool:"
# models = [
#     "This model can achieve more fine-grained vision perception for images with text and answer questions based on the reference OCR token.", # TextVQA
#     "This model is a state-of-the-art object detector that can output the object coordinates in images.", # Grounding 
#     "This model can answer questions about science, including natural science, language science, and social science.", # ScienceQA
#     "This model can answer visual questions from People Who Are Blind. When the provided information is insufficient, respond with 'Unanswerable'.", # VizWiz
#     "This model can give short answers to conduct visual reasoning of real scenes.", # GQA
# ]
models=[
    "This model is a financial expert and can deal with stocks problems in Chinese based on the provided candlestick chart. This model demonstrates exceptional proficiency in describing, question answering and trend prediction of the candlestick charts " # Fin
    "This model is an expert in Sciense, especially in biology, map understanding, physics, chemistry etc. However, it encounters challenges in understanding medical images." # Sci
    "This model is an expert in Medical image understangding, mainly focused on pathology, including pathology/cell sections and some natural pictures of the conditions." # Med
    "This model is an expert in ego view autonomous driving scene understanding, including prediction the coordinates, planning the next action, etc.", # AD
    "This model is an expert in understanding the remote sensing data, it shows strong ability in counting, checking the presence of a object, asking the area of the objects, etc." # RS
]
# options = ["1", "2", "3", "4", "5"]
options = ["A", "B", "C", "D", "E"]
for i in range(len(models)):
    selection_prompt = selection_prompt + "\n" + options[i] + ". " + models[i]
ROUTING_PROMPT = selection_prompt + "\n\nQuestion:\n"
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
        print("args.result_folders:", args.result_folders)
        self.result_folders = args.result_folders
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.names = "Fin Sci Med AD RS".split(' ')
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

    def __getitem__(self, index):
        line = self.questions[index]
        # try:
        qs_text = line["text"] #+ "\nAnswer:"#.replace("Answer with the option's letter from the given choices directly.","")
        ans = line['ans']
        # qs = qs + "\nYou can only choice answers shown below:\n"
        # qs = qs + "\n".join(ans) #+ "\nAnswer with the option's letter from the given choices directly."
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
    if sum(1 for c in routing_outputs if c not in 'ABCDE') > 1:
        print(f'[Warning] Routing outputs {routing_outputs} are not legal')
        assert False
    if 'A' in routing_outputs:
        return ans_candidates[0]
    elif 'B' in routing_outputs:
        return ans_candidates[1]
    elif 'C' in routing_outputs:
        return ans_candidates[2]
    elif 'D' in routing_outputs:
        return ans_candidates[3]
    elif 'E' in routing_outputs:
        return ans_candidates[4]
    else:
        print(f'[Warning] Routing outputs {routing_outputs} are not legal')
        assert False

def choose_ans_num(routing_outputs, ans_candidates):
    # check the rounting outputs is legal
    if sum(1 for c in routing_outputs if c not in '12345') > 1:
        print(f'[Warning] Routing outputs {routing_outputs} are not legal')
        assert False
    if '1' in routing_outputs:
        return ans_candidates[0]
    elif '2' in routing_outputs:
        return ans_candidates[1]
    elif '3' in routing_outputs:
        return ans_candidates[2]
    elif '4' in routing_outputs:
        return ans_candidates[3]
    elif '5' in routing_outputs:
        return ans_candidates[4]
    else:
        print(f'[Warning] Routing outputs {routing_outputs} are not legal')
        assert False

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

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-folders', type=str, default='results/CoIN/LLaVA')
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
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
    args = parser.parse_args()

    eval_model(args)
