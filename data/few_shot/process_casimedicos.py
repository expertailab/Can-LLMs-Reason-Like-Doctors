import os
import sys
import json

from pathlib import Path
from datasets import load_dataset

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from src.utils.json_funcs import append_to_jsonl



solution_idx_to_letter = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E"
}

processed_json_data = []
dataset = load_dataset("HiTZ/casimedicos-exp", "en")["train"]

for json_sample in dataset:
    idx_options = json_sample["options"]
    letter_options = {}
    for k, v in idx_options.items():
        if v == None:
            idx_options[k] = "None of them"
        letter_options[solution_idx_to_letter[k]] = v

    processed_json_data.append(
        {
            "id": json_sample["id"],
            "question": json_sample["full_question"],
            "options": letter_options,
            "answer": idx_options[str(json_sample["correct_option"])],
            "full_answer": json_sample["full_answer"],
            "answer_idx": solution_idx_to_letter[str(json_sample["correct_option"])]
        }
    )
        
processed_json_data = sorted(processed_json_data, key=lambda d: d["id"])
processed_json_data[0]

jsonl_file = '../data/few_shot/casimedicos.jsonl'

for sample_to_write in processed_json_data:
    append_to_jsonl(sample_to_write, jsonl_file)