# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import json
import numpy as np


class NerDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length, max_target_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_seq_length = self.max_source_length + self.max_target_length

        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    text = json_line["text"]
                    label = json_line["label"]
                    label = json.dumps(label, ensure_ascii=False)
                    self.data.append({
                        "text": text,
                        "label": label
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, text, label):
        messages = [
            {"role": "system",
             "content": "你的任务是做Ner任务提取, 根据用户输入提取出完整的实体信息, 并以JSON格式输出。"},
            {"role": "user", "content": text}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False, max_length=self.max_source_length,
                                     padding="max_length", pad_to_max_length=True, truncation=True)
        response = self.tokenizer(label, add_special_tokens=False, max_length=self.max_target_length,
                                  padding="max_length", pad_to_max_length=True, truncation=True)
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        attention_mask = (instruction["attention_mask"] + response["attention_mask"] + [1])
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]

        input_ids, attention_mask, labels = self.preprocess(**item_data)

        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)
        
