# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm

def lcs(X, Y):
    """Compute the length of the longest common subsequence of strings X and Y"""
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]

def rouge_l(list1, list2):
    """Compute ROUGE-L scores for pairs of strings in two lists"""
    scores = []
    for str1, str2 in zip(list1, list2):
        if len(str2.split()) == 0:  # 如果参考答案为空，赋予最低分数
            rouge_l_score = 0.0
        else:
            lcs_length = lcs(str1.split(), str2.split())
            rouge_l_score = lcs_length / len(str2.split())
        scores.append(rouge_l_score)
    
    average_rouge_l_score = sum(scores) / len(scores)  # 计算平均分数
    return average_rouge_l_score


def main():
    model_path = "qwen2.5/0.5B-instruct"
    #train_model_path = "output_ner"
    #train_model_path = "model/Qwen2.5-0.5B-Instruct"
    train_model_path = "qwen2.5/1.5B-sft-dolly"
    data_path = "dolly_new/val.json"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_case = []
    answers = []
    responses = []

    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            json_line = json.loads(line)
            ask = json_line["text"]
            test_case.append(ask)
            answer = json_line["label"]
            answers.append(answer)
    #print(test_case)
    """
    test_case = [
        "Which of Shakespeare’s plays is the longest?",
        "From the passage provided, extract the programming languages supported by Flink. Separate them with a comma.",
        "Which is a species of fish? Nurse or Nurse shark"
    ]
    """
    for case in tqdm(test_case):
        messages = [
            {"role": "system",
             "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request."},
            {"role": "user", "content": case}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=140,
            top_k=1
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses.append(response)
        #print("----------------------------------")
        #print(f"input: {case}\nresult: {response}")

    rougel_score = rouge_l(answers, responses)
    print(f"rouge_l score: {rougel_score}")


if __name__ == '__main__':
    main()

