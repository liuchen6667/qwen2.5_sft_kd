import json

def trans(file_path, save_path):
    with open(save_path, "a", encoding="utf-8") as w:
        with open(file_path, "r", encoding="utf-8") as r:
            for line in r:
                line = json.loads(line)
                text = line['text']
                label = line['label']
                trans_label = {}
                for key, items in label.items():
                    items = items.keys()
                    trans_label[key] = list(items)
                trans = {
                    "text": text,
                    "label": trans_label
                }
                line = json.dumps(trans, ensure_ascii=False)
                #print(line)
                w.write(line + "\n")
                w.flush()

if __name__ == '__main__':
    trans("ner_data_origin/train.json", "ner_data/train.json")
    trans("ner_data_origin/dev.json", "ner_data/val.json")

