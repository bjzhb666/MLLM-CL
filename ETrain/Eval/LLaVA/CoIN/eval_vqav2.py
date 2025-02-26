import os
import argparse
import json
import re

from ETrain.Eval.LLaVA.CoIN.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./playground/Instructions_slim/VQAv2/val.json')
    parser.add_argument('--result-file', type=str, default='./results/CoIN_slim/VQAv2/Zero_shot/merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def filter_string(input_str):
    result = re.sub(r'[^a-zA-Z0-9\s]', '', input_str)
    return result

def eval_single(annotation_file, result_file):
    annotations = json.load(open(annotation_file))
    annotations = {str(annotation['question_id']): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    pred_list = {
        "correct": [],
        "incorrect": []
    }
    for result in results:
        annotation = annotations[str(result['question_id'])]
        pred = result['text']
        ground_truth = annotation['answer']
        if filter_string(pred.upper()) == filter_string(ground_truth.upper()):
            right += 1
            pred_list['correct'].append({'pred': pred, 'gt': ground_truth})
        else:
            pred_list['incorrect'].append({'pred': pred, 'gt': ground_truth})

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))
        output_file = os.path.join(args.output_dir, 'Result.json')
        with open(output_file, 'w') as f:
            json.dump(pred_list, f)

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
