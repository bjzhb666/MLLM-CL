import os
import argparse
import json
import re
from evaluate import load



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default='./LLaVA/playground/Instructions_slim/OCRVQA/test_1.json')
    parser.add_argument('--result-file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/merge.jsonl')
    parser.add_argument('--output-dir', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune')
    return parser.parse_args()



def eval_single(annotation_file, result_file):
    bertscore = load("bertscore", module_type="metric", cache_dir="metrics")
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    annotations = json.load(open(annotation_file))
    annotations = {data['question_id']: data for data in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    total = len(results)
    right = 0
    for result in results:
        annotation = annotations[result['question_id']]
        ground_truth = annotation['answer']
        score = bertscore.compute(predictions=annotation, references=ground_truth, lang='zh')['f1']
        right += score
        pred_list.append(dict(
            question=annotation['question'],
            annotation=annotation,
            ground_truth=ground_truth,
            score=score,
        ))
        # if 'Unanswerable' in result['text'] :
        #     continue
        # if result['text'].lower() == ground_truth.lower():
        #     right += 1

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * right / total))
    #将结果写入文件
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * right / total))
        output_file = os.path.join(args.output_dir, 'Result.json')
        with open(output_file, 'w') as f:
            json.dump(pred_list, f)

if __name__ == "__main__":
    # list_evaluation_modules()
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)
