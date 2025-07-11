import os
import sys
import argparse

from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import evaluate
import numpy as np
import pandas as pd

from src.utils.preprocessing import trim_first_segment

from typing import Union
from datasets import (
    load_dataset,
    ClassLabel
)


DATA_PATH = "../data"
accuracy_metric = evaluate.load("accuracy")


id2llm = {
    "llama": [
        # "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct"
    ],
    "gemma": [
        # "google/gemma-3-1b-it",
        # "google/gemma-3-4b-it",
        # "google/gemma-3-12b-it",
        "google/gemma-3-27b-it"
    ],
    "phi": [
        # "microsoft/Phi-4-mini-instruct",
        # # "microsoft/Phi-3-small-8k-instruct",
        # "microsoft/phi-4",
        # "microsoft/Phi-3.5-MoE-instruct",

        # "microsoft/Phi-4-mini-reasoning",
        "microsoft/Phi-4-reasoning",
        # "microsoft/Phi-4-reasoning-plus"
    ],
    "olmo": [
        # "allenai/OLMo-2-0425-1B-Instruct",
        # "allenai/OLMo-2-1124-7B-Instruct",
        # "allenai/OLMo-2-1124-13B-Instruct",
        "allenai/OLMo-2-0325-32B-Instruct"
    ],
	"qwen": [
		# "Qwen/Qwen3-0.6B", 
		# "Qwen/Qwen3-1.7B",
		# "Qwen/Qwen3-4B",
		# "Qwen/Qwen3-8B",
		# "Qwen/Qwen3-14B",
		"Qwen/Qwen3-32B",
		# "Qwen/Qwen3-30B-A3B",
	],
	"deepseek": [
		# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
		# "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
		# "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
		# "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
		# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
		"deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
	],
	"deepcogito": [
		# "deepcogito/cogito-v1-preview-llama-3B",
		# "deepcogito/cogito-v1-preview-llama-8B",
		# "deepcogito/cogito-v1-preview-qwen-14B",
		# "deepcogito/cogito-v1-preview-qwen-32B",
		"deepcogito/cogito-v1-preview-llama-70B"
	],
    "mimo": [
        # "XiaomiMiMo/MiMo-7B-Base",
        # "XiaomiMiMo/MiMo-7B-RL-Zero",
        "XiaomiMiMo/MiMo-7B-SFT",
        # "XiaomiMiMo/MiMo-7B-RL"
    ],
    "medgemma": [
        # "google/medgemma-4b-it",
        "google/medgemma-27b-text-it"
    ],
    "meditron": [
        # "OpenMeditron/Meditron3-Gemma2-2B",
        # "OpenMeditron/Meditron3-Qwen2.5-7B",
        # "OpenMeditron/Meditron3-8B",
        # "OpenMeditron/Meditron3-Gemma2-9B",
        # "OpenMeditron/Meditron3-Qwen2.5-14B",
        # "OpenMeditron/Meditron3-Phi4-14B",
        "OpenMeditron/Meditron3-70B"
    ],
	"aloe": [
		# "HPAI-BSC/Qwen2.5-Aloe-Beta-7B",
		# "HPAI-BSC/Llama3.1-Aloe-Beta-8B",
		# "HPAI-BSC/Llama3.1-Aloe-Beta-70B",
		"HPAI-BSC/Qwen2.5-Aloe-Beta-72B"
	],
	"huatuo": [
		# "FreedomIntelligence/HuatuoGPT-o1-7B",
		# "FreedomIntelligence/HuatuoGPT-o1-8B",
		# "FreedomIntelligence/HuatuoGPT-o1-70B",
		"FreedomIntelligence/HuatuoGPT-o1-72B"
	],
    "gpt": [
        "ChatGPT/gpt-4o-2024-08-06"
    ],
    "open_router": [
        # "open_router/deepseek/deepseek-chat-v3-0324",
        # "open_router/deepseek/deepseek-r1-0528",
        "open_router/google/gemini-2.5-pro",
        # "open_router/anthropic/claude-sonnet-4",
        # "open_router/meta-llama/llama-4-maverick"
    ],
    "gemini": [
        "gemini/gemini-2.5-pro"
    ],
}


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


thresholds = {
    "medbullets": 17.2,
    "medexqa": 24.3,
    "medmcqa": 29.7,
    "medqa": 23.0,
    "medxpertqa-r": 11.0,
    "medxpertqa-u": 9.7,
    "mmlu": 29.2,
    "mmlu-pro": 11.7,
    "pubmedqa": 30.0,
    "medarc": 23.0,
    "metamedqa": 18.7,
    "average": 20.7
}


def load_bench_dataset(benchmark: str) -> Union[str, list]:
    dataset_path = f"{DATA_PATH}/benchmarks/{benchmark}"
    if benchmark == "medagentsbench":
        names = []
        for filename in os.listdir(dataset_path):
            names.append(filename.split(".")[0])
        return names
    if benchmark == "medarc":
        return ["medarc"]
    if benchmark == "metamedqa":
        return ["metamedqa"]
    

def evaluating(
    args: argparse.Namespace,
    predictions_path: str,
    predictions_file: str
) -> dict:
    all_acc_values = []
    max_correct = []
    evaluation_values = {"evaluation_values": {}}
    for seed in args.seeds:
        dataset = load_dataset("json", data_dir=DATA_PATH, data_files=f"{predictions_path}/seed_{seed}/{predictions_file}")["train"]
        dataset = dataset.filter(lambda x: x["similarity"]["label"] in ["similar", "different"])
        labels = ClassLabel(names=["similar", "different"])
        predictions = [prediction["label"] for prediction in dataset["similarity"]]
        accuracy = accuracy_metric.compute(
            references=labels.str2int(len(dataset)*["similar"]), 
            predictions=labels.str2int(predictions)
        )["accuracy"]*100
        all_acc_values.append(accuracy)
        evaluation_values["evaluation_values"][f"seed_{seed}"] = round(accuracy, 1)

        if len(max_correct) == 0:
            max_correct = [0] * len(dataset)

        predictions = [prediction["label"] for prediction in dataset["similarity"]]
        for idx, pred in enumerate(predictions):
            if pred == "similar":
                max_correct[idx] = 1

    evaluation_values["evaluation_values"]["avg"] = round(np.mean(all_acc_values), 1)
    evaluation_values["evaluation_values"]["std"] = round(np.std(all_acc_values), 2)
    evaluation_values["evaluation_values"]["upper_bound"] = round(len([x for x in max_correct if x])/len(max_correct)*100, 2)
    return evaluation_values


def color_cell(val: float, threshold: float) -> str:
    color = 'gray!15' if val == threshold else ('red!15' if val < threshold else 'green!15')
    return f'\\cellcolor{{{color}}}{val:.1f}'


def main(args: argparse.ArgumentParser) -> None:
    for answer_type in args.answer_types:
        for option in args.options:
            results = {"results": []}
            print(f"Answer Type: {answer_type} | Options: {option}")
            for llm_suite in args.llms:
                for llm_name in id2llm[llm_suite]:
                    for prompt in args.prompts:
                        result = {
                            "llm": trim_first_segment(llm_name),
                            "prompt": prompt,
                        }
                        collected_evaluation_values = []
                        collected_upper_bounds = []
                        collected_stds = []
                        for benchmark in args.benchmarks:
                            dataset_ids = load_bench_dataset(benchmark)
                            sorted_ids = sorted(dataset_ids)
                            for dataset in sorted_ids:
                                dataset_results = result.copy()
                                dataset_results["dataset"] = dataset
                                if dataset in ["medarc", "metamedqa"]:
                                    predictions_path = os.path.join(
                                        "predictions", benchmark, trim_first_segment(llm_name),
                                        f"answer_{answer_type}", f"option_{option}"
                                    )
                                else:
                                    predictions_path = os.path.join(
                                        "predictions", benchmark, dataset, trim_first_segment(llm_name),
                                        f"answer_{answer_type}", f"option_{option}"
                                    )
                                predictions_file = f"{prompt}_thinking.jsonl" if args.enable_thinking else f"{prompt}.jsonl"
                                print(f"Found predictions. Path: {predictions_path} | File: {predictions_file}")
                                evaluation_values = evaluating(
                                    args=args,
                                    predictions_path=predictions_path,
                                    predictions_file=predictions_file
                                )
                                dataset_results.update(evaluation_values)
                                results["results"].append(dataset_results)
                                collected_evaluation_values.append(evaluation_values["evaluation_values"]["avg"])
                                collected_upper_bounds.append(evaluation_values["evaluation_values"]["upper_bound"])
                                collected_stds.append(evaluation_values["evaluation_values"]["std"])
                        prompt_results = result.copy()
                        prompt_results["evaluation_values"] = {}
                        avg_accuracy = round(sum(collected_evaluation_values) / len(collected_evaluation_values), 1)
                        avg_upper_bound = round(sum(collected_upper_bounds) / len(collected_upper_bounds), 1)
                        avg_stds = round(sum(collected_stds) / len(collected_stds), 1)
                        prompt_results["dataset"] = "average"
                        prompt_results["evaluation_values"]["collected_avgs"] = collected_evaluation_values
                        prompt_results["evaluation_values"]["avg"] = avg_accuracy
                        prompt_results["evaluation_values"]["upper_bound"] = avg_upper_bound
                        prompt_results["evaluation_values"]["std"] = avg_stds
                        results["results"].append(prompt_results)

            if len(args.prompts) > 1:
                records = []
                for item in results["results"]:
                    llm = item["llm"]
                    prompt = item["prompt"]
                    dataset = item["dataset"]
                    accuracy = item["evaluation_values"].get("avg")
                    records.append((llm, prompt, dataset, accuracy))

                df = pd.DataFrame(records, columns=["llm", "prompt", "dataset", "accuracy"])
                pivot_df = df.pivot(index=["llm", "prompt"], columns=["dataset"], values="accuracy")
                models = [
                    trim_first_segment(model)
                    for key in args.llms
                    for model in id2llm.get(key, [])
                ]
                prompts = args.prompts
                row_order = [(model, prompt) for model in models for prompt in prompts]
                new_df = pivot_df.reindex(index=row_order, columns=column_order)
                # new_df["upper"] = [item["evaluation_values"].get("upper_bound") for item in results["results"] if item["dataset"] == "average"]
                # new_df["std"] = [item["evaluation_values"].get("std") for item in results["results"] if item["dataset"] == "average"]

                latex_df = new_df.copy()
                for col in column_order:
                    latex_df[col] = latex_df[col].apply(lambda x: color_cell(x, thresholds[col]) if pd.notnull(x) else '')
                print(latex_df.to_latex(header=True, index=True, float_format="%.1f"))
            else:
                records = []
                for item in results["results"]:
                    family = ""
                    llm = item["llm"]
                    dataset = item["dataset"]
                    accuracy = item["evaluation_values"].get("avg")
                    records.append((family, llm, dataset, accuracy))

                df = pd.DataFrame(records, columns=["family", "llm", "dataset", "accuracy"])
                pivot_df = df.pivot(index=["family", "llm"], columns=["dataset"], values="accuracy")
                models = [
                    trim_first_segment(model)
                    for key in args.llms
                    for model in id2llm.get(key, [])
                ]
                family = [""] * len(results)
                row_order = [(fam, model) for fam in family for model in models]
                new_df = pivot_df.reindex(index=row_order, columns=column_order)

                latex_df = new_df.copy()
                for col in column_order:
                    latex_df[col] = latex_df[col].apply(lambda x: color_cell(x, thresholds[col]) if pd.notnull(x) else '')
                print(latex_df.to_latex(header=True, index=True, float_format="%.1f"))

                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llms", 
        nargs="+", 
        required=True, 
        help="List of language model suits to use."
    )
    parser.add_argument(
        "--benchmarks", 
        nargs="+", 
        required=True, 
        choices=["medagentsbench", "medarc", "metamedqa"],
        help="List of datasets to process."
    )
    parser.add_argument(
        "--answer_types", 
        nargs="+", 
        default=["closed"], 
        choices=["open", "closed"],
        help="Expected answer type(s): 'open' for free-form, 'closed' for fixed choices."
    )
    parser.add_argument(
        "--options", 
        nargs="+", 
        default=["free"],
        choices=["letters", "free"],
        help="List to specify option types: 'letters' for indexed choices (e.g., A, B, C) and 'free' for unindexed options."
    )
    parser.add_argument(
        "--seeds", 
        nargs="+", 
        type=int,
        default=[0, 32, 64],
        help="Seeds for experiment runs."
    )
    parser.add_argument(
        "--prompts", 
        nargs="+", 
        default=["base"],
        choices=["base", "cot", "ir", "ap", "arr", "few_shot", "multi_persona"],
        help="Prompting strategies to use."
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activate thinking mode (True or False)"
    )
    args = parser.parse_args()
    main(args)
