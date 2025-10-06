import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from src.utils.preprocessing import trim_first_segment


DATA_PATH = "../data"
ANNOTATIONS_PATH = "../data/fine_grained"


id2llm = {
    "random": [
        "random/random-baseline"
    ],
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
        "ChatGPT/gpt-4o-2024-08-06",
        "ChatGPT/o3-2025-04-16"
    ],
    "open_router": [
        "open_router/deepseek/deepseek-chat-v3-0324",
        "open_router/deepseek/deepseek-r1-0528",
        "open_router/anthropic/claude-sonnet-4",
        "open_router/meta-llama/llama-4-maverick"
    ],
    "gemini": [
        "gemini/gemini-2.5-pro"
    ],
}


datasets = [
	'medbullets',
	'medqa',
	'medxpertqa-r',
	'medxpertqa-u',
	'mmlu',
    'mmlu-pro',
	'pubmedqa',
]


reasoning_types = [
	'abduction-deduction',
	'abduction',
	'deduction',
	'induction',
]


def compute_accuracy(lst):
    return round(sum(lst) / len(lst) if lst else 0, 2)


def evaluating(
    llm_name: str,
    answer_type: str,
    option: str,
    prompt: str
) -> dict:
    result = {}
    llm_label = llm_name.split("/")[-1]
    result[llm_label] = {key: [] for key in reasoning_types}
    annotation_file = json.loads(Path(f"{ANNOTATIONS_PATH}/reasoning_types.json").read_text(encoding="utf-8"))
    for dataset in datasets:
        if llm_label == "random-baseline":
            predictions_path = os.path.join(
                DATA_PATH, "predictions", "medagentsbench", dataset, trim_first_segment(llm_name),
                "seed_0", "random.jsonl"
            )
        else:
            predictions_path = os.path.join(
                DATA_PATH, "predictions", "medagentsbench", dataset, trim_first_segment(llm_name),
                f"answer_{answer_type}", f"option_{option}", "seed_0", f"{prompt}_thinking.jsonl" if "cogito" in llm_name else f"{prompt}.jsonl"
            )
        predictions = [json.loads(line) for line in Path(predictions_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        dataset_subset = annotation_file[dataset]
        for reasoning_strategy, data_samples in dataset_subset.items():
            for prediction in predictions:
                if str(prediction["id"]) in data_samples:
                    label2id = 1 if prediction["similarity"]["label"] == "similar" else 0
                    result[llm_label][reasoning_strategy].append(label2id)
    for _, tasks in result.items():
        for task, values in tasks.items():
            tasks[task] = compute_accuracy(values)
    
    return result


def main(args: argparse.Namespace) -> None:
    for answer_type in args.answer_types:
        for option in args.options:
            model_accuracies = {}
            for llm_suite in args.llms:
                for llm_name in id2llm[llm_suite]:
                    for prompt in args.prompts:
                        evaluation_values = evaluating(
                            llm_name=llm_name,
                            answer_type=answer_type,
                            option=option,
                            prompt=prompt
                        )
                        model_accuracies.update(evaluation_values)

    # # Task labels
    # tasks = ['abduction-deduction', 'abduction', 'deduction', 'induction']
    # num_tasks = len(tasks)
    # angles = np.linspace(0, 2 * np.pi, num_tasks, endpoint=False).tolist()
    # angles += angles[:1]  # complete the loop

    # # Prepare the figure
    # num_models = len(model_accuracies)
    # cols = 3
    # rows = int(np.ceil(num_models / cols))
    # plt.figure(figsize=(cols*5, rows*5))

    # for idx, (model, scores) in enumerate(model_accuracies.items(), 1):
    #     values = [scores[task] for task in tasks]
    #     values += values[:1]  # complete the loop
        
    #     ax = plt.subplot(rows, cols, idx, polar=True)
    #     ax.plot(angles, values, 'o-', linewidth=2, label=model)
    #     ax.fill(angles, values, alpha=0.25)
    #     ax.set_xticks(angles[:-1])
    #     ax.set_xticklabels(tasks)
    #     ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    #     ax.set_ylim(0, 1)
    #     ax.set_title(model, fontsize=10)

    # plt.tight_layout()
    # plt.savefig("all_models_radar_chart.png")
    # plt.close()

    tasks = ['abduction-deduction', 'abduction', 'deduction', 'induction']
    models = [model_name.split("/")[-1] for model_name in list(model_accuracies.keys())]
    num_models = len(models)
    num_tasks = len(tasks)

    # X positions for groups
    x = np.arange(num_tasks)
    width = 0.7 / num_models  # width of each bar

    # Generate a color map
    colors = plt.get_cmap('Paired', num_models)

    plt.figure(figsize=(18.5, 8))

    # Plot bars
    for i, model in enumerate(models):
        accuracies = [model_accuracies[model][task] for task in tasks]
        plt.bar(x + i*width, accuracies, width=width, label=model, color=colors(i))

    plt.xticks(x + width*num_models/2, tasks, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Task', fontsize=16, labelpad=30)
    plt.ylabel('Accuracy', fontsize=16, labelpad=30)
    plt.ylim(0, 1)

    smaller_models = [
        'cogito-v1-preview-llama-70B',
        'Phi-4-reasoning',
        'medgemma-27b-text-it',
        'HuatuoGPT-o1-72B'
    ]

    bigger_models = [
        'deepseek-chat-v3-0324',
        'deepseek-r1-0528',
        'claude-sonnet-4',
        'llama-4-maverick',
        'gemini-2.5-pro',
        'gpt-4o-2024-08-06',
        'o3-2025-04-16'
    ]

    random_model = ['random-baseline']

    # Get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Frontier models
    bigger_handles = [handles[models.index(m)] for m in bigger_models]
    bigger_labels = bigger_models
    legend1 = plt.legend(bigger_handles, bigger_labels, title='Frontier Models',
                        loc='lower center', bbox_to_anchor=(0.5, 1.05),
                        ncol=len(bigger_labels), fontsize=13, title_fontsize=15)
    plt.gca().add_artist(legend1)

    # Small/Mid models
    smaller_handles = [handles[models.index(m)] for m in smaller_models]
    smaller_labels = smaller_models
    legend2 = plt.legend(smaller_handles, smaller_labels, title='Small/Mid Models',
                        loc='lower center', bbox_to_anchor=(0.6, 1.20),
                        ncol=len(smaller_labels), fontsize=13, title_fontsize=15)
    plt.gca().add_artist(legend2)

    # Random baseline
    random_handles = [handles[models.index(random_model[0])]]
    random_labels = random_model
    plt.legend(random_handles, random_labels, title='Random',
            loc='lower center', bbox_to_anchor=(0.15, 1.20),
            ncol=len(random_labels), fontsize=13, title_fontsize=15)

    plt.tight_layout()
    plt.savefig(f"{ANNOTATIONS_PATH}/fine_grained_bar_plot.png")
    plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llms", 
        nargs="+", 
        required=True, 
        help="List of language model suits to use."
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
        default=[0],
        help="Seeds for experiment runs."
    )
    parser.add_argument(
        "--prompts", 
        nargs="+", 
        default=["base"],
        choices=["base", "cot", "ir", "ap", "arr", "few_shot", "multi_persona"],
        help="Prompting strategies to use."
    )
    args = parser.parse_args()
    main(args)