import json
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qf", type=str, default="sqa")
    parser.add_argument("--answers-file", type=str, default='results/CoIN/LLaVA')
    args = parser.parse_args()

    agent_selection_count = {}

    answers_file = args.answers_file  # 替换为你的答案文件路径

    # GT_map={
    #     'sqa': 'C',
    #     'GQA': 'E',
    #     'TextVQA': 'A',
    #     'VizWiz': 'D',
    #     'Grounding': 'B',
    # }
    GT_map={
        'fin':'A',
        'sci':'B',
        'pathvqa':'C',
        'drivelm':'D',
        'rs':'E'
    }
    with open(answers_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if 'agent_selection' in data:
                routing_outputs = data['agent_selection']

            if routing_outputs in agent_selection_count:
                agent_selection_count[routing_outputs] += 1
            else:
                agent_selection_count[routing_outputs] = 1

    # TODO:统计选择正确率
    task_name = args.qf
    # 统计ans_file中agent_selection的答案
    GT_choose = GT_map[task_name]
    # 统计GT_choose在agent_selection_count中的数量占比
    if GT_choose is not None:
        GT_count = agent_selection_count.get(GT_choose, 0)
        total_count = sum(agent_selection_count.values())
        # calculate the accuracy
        GT_percentage = GT_count / total_count *100
        print(f'[Info] {task_name} GT_choose: {GT_choose}, GT_count: {GT_count}, total_count: {total_count}, GT_percentage: {GT_percentage}%')