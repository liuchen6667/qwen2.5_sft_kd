import json

def trans(file_path, save_path):
    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['instruction']
                label = line['output']
                
                trans = {
                    "text": text,
                    "label": label
                }
                line = json.dumps(trans, ensure_ascii=False)
                print(line)
                w.write(line + "\n")
                w.flush()

if __name__ == '__main__':
    trans("dolly/raw.jsonl", "dolly_new/train.json")
    trans("dolly/valid.jsonl", "dolly_new/val.json")

