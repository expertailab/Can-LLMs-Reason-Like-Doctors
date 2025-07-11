import os
import sys
import argparse

from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import evaluate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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
        "microsoft/phi-4",
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
	]
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

pt2upper = {
    "base": "Base",
    "cot": "CoT",
    "ir": "IR",
    "ap": "AP",
    "few_shot": "FS",
    "multi_persona": "MP"
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


def main(args: argparse.ArgumentParser) -> None:
    if len(args.prompts) > 1:
        for answer_type in args.answer_types:
            for option in args.options:
                data = []
                for llm_suite in args.llms:
                    for llm_name in id2llm[llm_suite]:
                        model = llm_name.split("/")[1]
                        for prompt in args.prompts:
                            for benchmark in args.benchmarks:
                                dataset_ids = load_bench_dataset(benchmark)
                                sorted_ids = sorted(dataset_ids)
                                for dataset in sorted_ids:
                                    if dataset in ["medarc", "metamedqa"]:
                                        predictions_path = os.path.join(
                                            "predictions", benchmark, llm_name.split("/")[1],
                                            f"answer_{answer_type}", f"option_{option}"
                                        )
                                    else:
                                        predictions_path = os.path.join(
                                            "predictions", benchmark, dataset, llm_name.split("/")[1],
                                            f"answer_{answer_type}", f"option_{option}"
                                        )
                                    if llm_suite in ["qwen", "deepcogito"]:
                                        predictions_files = [f"{prompt}.jsonl", f"{prompt}_thinking.jsonl"]
                                    else:
                                        predictions_files = [f"{prompt}.jsonl"]
                                    for predictions_file in predictions_files:
                                        print(f"Found predictions. Path: {predictions_path} | File: {predictions_file}")
                                        value = evaluating(
                                            args=args,
                                            predictions_path=predictions_path,
                                            predictions_file=predictions_file
                                        )["evaluation_values"]["avg"]
                                        if "_thinking" in predictions_file:
                                            data.append({
                                                'Model': f'{model}_thinking',
                                                'Prompt': f'{pt2upper[prompt]}',
                                                'Dataset': f'{dataset}',
                                                'Value': value
                                            })
                                        else:
                                            data.append({
                                                'Model': f'{model}',
                                                'Prompt': f'{pt2upper[prompt]}',
                                                'Dataset': f'{dataset}',
                                                'Value': value
                                            })

                df = pd.DataFrame(data)
                plt.figure(figsize=(18, 6))
                sns.boxplot(x='Model', y='Value', hue='Prompt', data=df)
                plt.ylabel('Accuracy Value (Aggregated over the 11 Datasets from the 3 Benchmarks.)')
                plt.xticks(rotation=45)
                prompts = df['Prompt'].unique()
                palette = sns.color_palette(n_colors=len(prompts))
                patches = [mpatches.Patch(color=palette[i], label=prompts[i]) for i in range(len(prompts))]
                plt.legend(handles=patches,
                        title='Prompt',
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        borderaxespad=0,
                        frameon=False,
                        fontsize=10,
                        title_fontsize=12)
                plt.subplots_adjust(right=0.85)
                plt.savefig(f'{DATA_PATH}/model_prompt_boxplot.png', dpi=300, bbox_inches='tight')
                plt.close()

    else:
        raise ValueError(
            f"No plot designed for less than 2 different prompts."
        )

                



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
        choices=["base", "cot", "ir", "ap", "few_shot", "multi_persona"],
        help="Prompting strategies to use."
    )
    args = parser.parse_args()
    main(args)
