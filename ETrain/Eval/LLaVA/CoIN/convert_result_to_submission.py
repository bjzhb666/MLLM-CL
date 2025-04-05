import json
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/merge.jsonl')
    parser.add_argument('--output_file', type=str, default='./LLaVA/results/CoIN_slim_new_0.8/OCRVQA/Finetune/our_result_for_submission.tsv')
    args=parser.parse_args()
    
    # 读取 TSV 文件
    tsv_file = 'llava_v1.5_7b_MMT-Bench_ALL_openai_submission.tsv'
    tsv_data = pd.read_csv(tsv_file, sep='\t')

    # 读取 JSONL 文件并提取需要的字段
    jsonl_file = args.result_file
    
    
    
    jsonl_data = {}

    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)
            question_id = entry['question_id'].replace("ALL_", "")
            text = entry['text'].strip().upper()  # 去除多余符号并转为大写
            jsonl_data[int(question_id)] = {"question_id": question_id, 'text': text}

    # 更新 TSV 数据
    for index, row in tsv_data.iterrows():
        # 查找匹配的 JSONL 条目
        matched_entry = jsonl_data.get(index, None)
        
        if matched_entry:
            print('match')
            # 更新 prediction 和 opt 字段
            tsv_data.at[index, 'prediction'] = row['prediction']  # 保持原值
            tsv_data.at[index, 'opt'] = matched_entry['text']  # 更新为 JSONL 中的 text
        else:
            # 没有对应的 index，设置为默认值
            tsv_data.at[index, 'prediction'] = "0"
            tsv_data.at[index, 'opt'] = 'Z'

    # 写回更新后的 TSV 文件
    tsv_data.to_csv(args.output_file, sep='\t', index=False)

    print("TSV file updated successfully!")