import json
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

def get_token_distribution(file_path, tokenizer):
    input_num_tokens, outout_num_tokens = [], []
    with open(file_path, "r", encoding="utf-8") as r:
        for line in r:
            line = json.loads(line)
            text = line['text']
            label = line['label']
            label = json.dumps(label, ensure_ascii=False)
            input_num_tokens.append(len(tokenizer(text).input_ids))
            outout_num_tokens.append(len(tokenizer(label).input_ids))
    return min(input_num_tokens), max(input_num_tokens), np.mean(input_num_tokens),\
        min(outout_num_tokens), max(outout_num_tokens), np.mean(outout_num_tokens)


def main():
    model_path = "model/Qwen2.5-0.5B-Instruct"
    train_data_path = "ner_data/train.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    i_min, i_max, i_avg, o_min, o_max, o_avg = get_token_distribution(train_data_path, tokenizer)
    print(i_min, i_max, i_avg, o_min, o_max, o_avg)

    plt.figure(figsize=(8, 6))
    bars = plt.bar([
        "input_min_token",
        "input_max_token",
        "input_avg_token",
        "ouput_min_token",
        "ouput_max_token",
        "ouput_avg_token",
    ], [
        i_min, i_max, i_avg, o_min, o_max, o_avg
    ])
    plt.title('训练集Token分布情况')
    plt.ylabel('数量')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, int(yval), va='bottom')
    plt.savefig('train_token_distribution.png')
    #plt.show()

if __name__ == '__main__':
    main()

