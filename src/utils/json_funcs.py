import json


def append_to_jsonl(data, file_path):
    with open(file_path, "a") as file:
        json.dump(data, file)
        file.write("\n")


def load_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data