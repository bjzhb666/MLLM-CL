import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default = './cl_dataset/ScienceQA')
    parser.add_argument('--result-file', type=str, default='./results/CoIN/Qwen/ScienceQA/Finetune/merge.jsonl')
    parser.add_argument('--output-file', type=str, default= './results/CoIN/Qwen/ScienceQA/Finetune/output.jsonl')
    parser.add_argument('--output-result', type=str, default= './results/CoIN/Qwen/ScienceQA/Finetune/output_result.jsonl')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return -1
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    # split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    # problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    # split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    for prob_id, pred in predictions.items():
        # pred = predictions[prob_id]
        pred_text = pred['text']
        answer = pred_text.strip()

        ds = [
            "GQA",
            "Grounding",
            "ImageNet",
            "OCRVQA",
            "ScienceQA",
            "TextVQA",
            "VizWiz",
            "VQAv2"
        ]
        dataname = base_dir.split('/')[-1]
        if dataname in ds:
            gt = str(ds.index(dataname))

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': gt,
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        # sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        if answer == gt:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])

    ###### IMG ######
    # multimodal_correct = len([x for x in results['correct'] if x['is_multimodal']])
    # multimodal_incorrect = len([x for x in results['incorrect'] if x['is_multimodal']])
    # multimodal_total = multimodal_correct + multimodal_incorrect
    ###### IMG ######

    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
