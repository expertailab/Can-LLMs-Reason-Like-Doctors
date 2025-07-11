import os
import json

from src.prompts.conversational import construct_prompt
from datasets import (
    load_dataset,
    DatasetDict,
    Dataset
)


DATA_PATH = "../data"


def trim_first_segment(s):
    return s.split('/', 1)[1] if '/' in s else ''


def extract_answer(example: dict):
    example["options"] = json.loads(example["options"])
    return {"answer": example["options"][example["answer_index"]]}


def load_benchmarks(benchmark_names: list) -> dict:
    benchmarks = {}
    for benchmark in benchmark_names:
        dataset_path = f"{DATA_PATH}/benchmarks/{benchmark}"
        if benchmark == "medagentsbench":
            names_data = []
            for filename in os.listdir(dataset_path):
                dataset = load_dataset("json", data_files=f"{dataset_path}/{filename}")["train"]
                dataset = dataset.rename_column("realidx", "id")
                dataset = dataset.filter(lambda x: x["answer_idx"] is not None)
                dataset.info.dataset_name = filename.split(".")[0]
                names_data.append({"name": filename.split(".")[0], "data": dataset})
            dataset = DatasetDict({name_data["name"]: name_data["data"] for name_data in names_data})
            benchmarks[benchmark] = dataset
        if benchmark == "medarc":
            dataset = load_dataset("json", data_files=f"{dataset_path}.jsonl")["train"]
            dataset = dataset.rename_column("question_id", "id")
            dataset = dataset.rename_column("answer", "answer_idx")
            dataset = dataset.filter(lambda x: x["answer_idx"] is not None)
            dataset = dataset.map(extract_answer)
            dataset.info.dataset_name = "medarc"
            benchmarks[benchmark] = dataset
        if benchmark == "metamedqa":
            dataset = load_dataset("json", data_files=f"{dataset_path}.jsonl")["train"]
            medqa_dataset = load_dataset("json", data_files=f"{DATA_PATH}/benchmarks/medagentsbench/medqa.jsonl")["train"]
            medqa_questions = set(medqa_dataset["question"])
            filtered_dataset = dataset.filter(lambda x: x["question"] in medqa_questions)
            filtered_dataset = filtered_dataset.map(lambda example, idx: {"id": idx}, with_indices=True)
            filtered_dataset = filtered_dataset.filter(lambda x: x["answer_idx"] is not None)
            filtered_dataset.info.dataset_name = "metamedqa"
            benchmarks[benchmark] = filtered_dataset
    return benchmarks


def construct_input(
    dataset: Dataset,
    prompting_strategy: str,
    answer_type: str,
    llm_name: str,
    role: str
) -> list:
    return [
        construct_prompt(
            dataset_name=dataset.info.dataset_name,
            answer_type=answer_type,
            data_sample=data_sample,
            prompting_strategy=prompting_strategy,
            llm_name=llm_name,
            role=role
        )
        for data_sample in dataset
    ]