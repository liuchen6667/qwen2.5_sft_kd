# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tqdm import tqdm



def main():
    model_path = "model/Qwen2.5-0.5B-Instruct"
    train_model_path = "output_ner"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    file_path = "dolly/raw.jsonl"
    save_path = "dolly_new/black_train.json"

    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in tqdm(r):
                line = json.loads(line)
                ask = line['instruction']
                #替换label
                #label = line['output']
                case = ask

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
                print("----------------------------------")
                print(f"input: {case}\nresult: {response}")

                label = response
                trans = {
                    "text": ask,
                    "label": label
                }
                line = json.dumps(trans, ensure_ascii=False)
                print(line)
                w.write(line + "\n")
                w.flush()


if __name__ == '__main__':
    main()

