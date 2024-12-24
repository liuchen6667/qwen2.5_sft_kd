# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def main():
    model_path = "qwen2.5/0.5B-instruct"
    train_model_path = "qwen2.5/0.5B-sft-dolly"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(train_model_path, trust_remote_code=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_case = [
        "Which of Shakespeare’s plays is the longest?",
        "From the passage provided, extract the programming languages supported by Flink. Separate them with a comma.",
        "Which is a species of fish? Nurse or Nurse shark"
    ]

    for case in test_case:
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


if __name__ == '__main__':
    main()

