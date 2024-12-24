# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from dolly_dataset import NerDataset
from tqdm import tqdm
import time, sys
import json

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

"""
def rouge_l(list1, list2):
    
    scores = []
    for str1, str2 in zip(list1, list2):
        lcs_length = lcs(str1.split(), str2.split())
        rouge_l_score = lcs_length / len(str2.split())
        scores.append(rouge_l_score)
        average_rouge_l_score = sum(scores) / len(scores)
    return average_rouge_l_score
"""

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

def evaluate_rouge_l(model, tokenizer, device, data_path):
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

    rougel_score = rouge_l(answers, responses)
    return rougel_score

def train_model(model, train_loader, val_loader, optimizer,
                device, num_epochs, model_output_dir, writer, tokenizer, val_json_path):
    best_rouge_l = 0.0
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss, batch_step)
            batch_step += 1
            # 100轮打印一次 loss
            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")

        # 计算ROUGE-L
        rouge_l_score = evaluate_rouge_l(model, tokenizer, device, val_json_path)
        print(f"ROUGE-L score: {rouge_l_score} , epoch: {epoch}")
        writer.add_scalar('ROUGE-L/val', rouge_l_score, epoch)

        # 保存最佳模型
        if rouge_l_score > best_rouge_l:
            best_rouge_l = rouge_l_score
            model.save_pretrained(model_output_dir)
            print(f"New best ROUGE-L score: {best_rouge_l}. Model saved.")
            print("Save Model To ", model_output_dir)

        print(f"best ROUGE-L score: {best_rouge_l}.")


def validate_model(model, device, val_loader):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
    model_name = "qwen2.5/3B-instruct"
    # 训练集
    train_json_path = "dolly_new/train.json"
    # 验证集
    val_json_path = "dolly_new/val.json"
    max_source_length = 50
    max_target_length = 140
    epochs = 30
    batch_size = 5
    lr = 1e-4
    model_output_dir = "qwen2.5/3B-sft-dolly"
    logs_dir = "logs"
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    print("Start Load Train Data...")
    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 4,
    }
    training_set = NerDataset(train_json_path, tokenizer, max_source_length, max_target_length)
    training_loader = DataLoader(training_set, **train_params)
    print("Start Load Validation Data...")
    val_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 4,
    }
    val_set = NerDataset(val_json_path, tokenizer, max_source_length, max_target_length)
    val_loader = DataLoader(val_set, **val_params)
    # 日志记录
    writer = SummaryWriter(logs_dir)
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    model = model.to(device)
    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer,
        tokenizer=tokenizer,
        val_json_path=val_json_path
    )
    writer.close()


if __name__ == '__main__':
    main()

