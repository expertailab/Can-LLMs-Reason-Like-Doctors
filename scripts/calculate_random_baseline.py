import os
import sys
import random
import argparse
import numpy as np
import evaluate
import pandas as pd

from pathlib import Path
from datasets import (
    load_dataset,
    Dataset,
    ClassLabel
)

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from src.utils.json_funcs import append_to_jsonl
from src.utils.preprocessing import load_benchmarks


DATA_PATH = "../data"
accuracy_metric = evaluate.load("accuracy")


column_order = [
    'medbullets',
    'medexqa',
    'medmcqa',
    'medqa',
    'medxpertqa-r',
    'medxpertqa-u',
    'mmlu',
    'mmlu-pro',
    'pubmedqa',
    'medarc',
    'metamedqa',
    'average'
]


row_order = [
    "random-baseline"
]


def process_options(example):
    if 'options' in example:
        if isinstance(example['options'], dict):
            example['options'] = list(example['options'].values())
    return example


def log_summary(
        action: str,
        count: int,
        model_name: str,
        dataset_name: str,
        seed: str
    ) -> None:
    print(
        f"{action} {count} solutions | "
        f"Model: {model_name} | "
        f"Dataset: {dataset_name} | "
        f"Seed: {seed}"
    )
    print('-' * 135)


def label_and_save_solutions(
        solutions: list[dict],
        output_path: str
    ) -> None:
    for sample in solutions:
        label = {"label": "different"}

        if sample["provided_solution"] is not None:
            if sample["provided_solution"] == sample["gold_solution"]:
                label = {"label": "similar"}

        sample["similarity"] = label
        append_to_jsonl(sample, output_path)


def select_random(options):
    random_choice = random.choice(options)
    return random_choice


def get_predictions(
        predictions_jsonl: str,
        dataset: Dataset,
        seed: str
    ) -> None:

    dataset = dataset.map(process_options)
    provided_solutions = []
    success_count = 0
    for sample in dataset:
        try:
            options = [option for option in sample["options"] if option]
            solution = select_random(options)
            success_count += 1
        except Exception as e:
            print(f"[Error parsing solution]: {e}")
            solution = None

        provided_solutions.append({
            "id": sample["id"],
            "provided_solution": solution,
            "gold_solution": sample["answer"]
        })

    log_summary("Provided", success_count, "random-baseline", dataset.info.dataset_name, seed)

    label_and_save_solutions(provided_solutions, predictions_jsonl)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_predictions_path(
        base_path: str,
        benchmark: str,
        seed: int,
        dataset_name: str = None
    ) -> str:
    parts = [base_path, "predictions", benchmark]
    if dataset_name:
        parts.append(dataset_name)
    parts.extend([
        "random-baseline", f"seed_{seed}"
    ])
    predictions_dir = os.path.join(*parts)
    ensure_dir(predictions_dir)
    return os.path.join(predictions_dir, f"random.jsonl")


def main(args: argparse.Namespace) -> None:
    model = "random-baseline"
    result_scheleton = {
        "results": []
    }
    collected_avgs = []
    collected_upper_bounds = []
    collected_stds = []
    benchmarks = load_benchmarks(benchmark_names=args.benchmarks)
    for benchmark in args.benchmarks:
        benchmark_path = Path(os.path.join(DATA_PATH, "predictions", benchmark))

        dataset_names = [None]
        if benchmark == "medagentsbench":
            dataset_names = [d.name for d in benchmark_path.iterdir()]

        for dataset_name in dataset_names:
            if dataset_name:
                dataset = benchmarks[benchmark][dataset_name]
            else:
                dataset = benchmarks[benchmark]

            result = {
                "model": model,
                "evaluation_values": {}
            }
            all_acc_values = []
            max_correct = []
            result["dataset"] = dataset.info.dataset_name

            for seed in args.seeds:
                predictions_jsonl = get_predictions_path(
                    base_path=DATA_PATH,
                    benchmark=benchmark,
                    seed=seed,
                    dataset_name=dataset_name
                )
                if not os.path.exists(predictions_jsonl):
                    get_predictions(
                        predictions_jsonl=predictions_jsonl,
                        dataset=dataset,
                        seed=seed
                    )
                labeled_dataset = load_dataset("json", data_files=predictions_jsonl)["train"]
                labeled_dataset = labeled_dataset.filter(lambda x: x["similarity"]["label"] in ["similar", "different"])
                labels = ClassLabel(names=["similar", "different"])
                predictions = [prediction["label"] for prediction in labeled_dataset["similarity"]]
                accuracy = accuracy_metric.compute(
                    references=labels.str2int(len(labeled_dataset)*["similar"]), 
                    predictions=labels.str2int(predictions)
                )["accuracy"]*100
                all_acc_values.append(accuracy)
                result["evaluation_values"][f"seed_{seed}"] = round(accuracy, 1)

                if len(max_correct) == 0:
                    max_correct = [0] * len(labeled_dataset)

                predictions = [prediction["label"] for prediction in labeled_dataset["similarity"]]
                for idx, pred in enumerate(predictions):
                    if pred == "similar":
                        max_correct[idx] = 1

            result["evaluation_values"]["avg"] = round(np.mean(all_acc_values), 1)
            result["evaluation_values"]["std"] = round(np.std(all_acc_values), 2)
            result["evaluation_values"]["upper_bound"] = round(len([x for x in max_correct if x])/len(max_correct)*100, 2)
            result_scheleton["results"].append(result)
            collected_avgs.append(result["evaluation_values"]["avg"])
            collected_upper_bounds.append(result["evaluation_values"]["upper_bound"])
            collected_stds.append(result["evaluation_values"]["std"])
                    
    result_scheleton["results"].append(
        {
            "model": "random-baseline",
            "evaluation_values": {
                "collected_avgs": collected_avgs,
                "avg": round(sum(collected_avgs) / len(collected_avgs), 1),
                "upper": round(sum(collected_upper_bounds) / len(collected_upper_bounds), 1),
                "std": round(sum(collected_stds) / len(collected_stds), 1)
            },
            "dataset": "average"
        }
    )

    records = []
    for item in result_scheleton["results"]:
        model = item["model"]
        dataset = item["dataset"]
        avg = item["evaluation_values"].get("avg")
        records.append((model, dataset, avg))

    df = pd.DataFrame(records, columns=["model", "dataset", "avg"])
    pivot_df = df.pivot(index="model", columns="dataset", values="avg")
    new_df = pivot_df.reindex(index=row_order, columns=column_order)
    # new_df["upper"] = [item["evaluation_values"].get("upper") for item in result_scheleton["results"] if item["dataset"] == "average"]
    # new_df["std"] = [item["evaluation_values"].get("std") for item in result_scheleton["results"] if item["dataset"] == "average"]
    print(new_df.to_latex(header=True, index=True, float_format="%.1f"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks", 
        nargs="+", 
        required=True, 
        choices=["medagentsbench", "medarc", "metamedqa"],
        help="List of datasets to process."
    )
    parser.add_argument(
        "--seeds", 
        nargs="+", 
        type=int,
        default=[0, 32, 64],
        help="Seeds for experiment runs."
    )
    args = parser.parse_args()
    main(args)