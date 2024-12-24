# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from dolly_dataset import NerDataset
from tqdm import tqdm
import time, sys


def train_model(model, train_loader, val_loader, optimizer,
                device, num_epochs, model_output_dir, writer):
    batch_step = 0
    for epoch in range(num_epochs):
        time1 = time.time()
        """
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
        """
        # 验证
        model.eval()
        val_loss = validate_model(model, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")
        #print("Save Model To ", model_output_dir)
        #model.save_pretrained(model_output_dir)


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
            #print(outputs.logits.shape)
            #print(outputs.keys())
            loss = outputs.loss
            running_loss += loss.item()
    return running_loss / len(val_loader)


def main():
    # 基础模型位置
    model_name = "model/Qwen2.5-0.5B-Instruct"
    train_model_path = "output_ner"
    # 训练集
    train_json_path = "dolly_new/train.json"
    # 验证集
    val_json_path = "dolly_new/val.json"
    max_source_length = 50
    max_target_length = 140
    epochs = 1
    batch_size = 5
    lr = 1e-4
    model_output_dir = "output_ner"
    logs_dir = "logs"
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
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
        writer=writer
    )
    writer.close()


if __name__ == '__main__':
    main()

