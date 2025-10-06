import os
import sys
import json

from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import evaluate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.preprocessing import trim_first_segment

MELD_PATH = "../data/meld"
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
	],
    "gpt": [
        # "ChatGPT/gpt-4o-2024-08-06",
        "ChatGPT/o3-2025-04-16"
    ],
    "open_router": [
        # "open_router/deepseek/deepseek-chat-v3-0324",
        "open_router/deepseek/deepseek-r1-0528",
        # "open_router/google/gemini-2.5-pro",
        "open_router/anthropic/claude-sonnet-4",
        "open_router/meta-llama/llama-4-maverick"
    ],
    "gemini": [
        "gemini/gemini-2.5-pro"
    ],
}


datasets = [
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
]


model_order = [
    "Llama-3.3-70B-Instruct",
    "gemma-3-27b-it",
    "phi-4",
    "Phi-4-reasoning",
    "OLMo-2-0325-32B-Instruct",
    "Qwen3-32B",
    "DeepSeek-R1-Distill-Llama-70B",
    "cogito-v1-preview-llama-70B",
    "MiMo-7B-SFT",
    "medgemma-27b-text-it",
    "Meditron3-70B",
    "Qwen2.5-Aloe-Beta-72B",
    "HuatuoGPT-o1-72B",
    "o3-2025-04-16",
    "deepseek-r1-0528",
    "claude-sonnet-4",
    "llama-4-maverick",
    "gemini-2.5-pro"
]


palette = {
    "Llama-3.3-70B-Instruct": "#1F77B4",
    "llama-4-maverick": "#005C99",
    "gemma-3-27b-it": "#FF7F0E",
    "medgemma-27b-text-it": "#FFBB78",
    "phi-4": "#2CA02C",
    "Phi-4-reasoning": "#98DF8A",
    "OLMo-2-0325-32B-Instruct": "#8C564B",
    "Qwen3-32B": "#9467BD",
    "Qwen2.5-Aloe-Beta-72B": "#C5B0D5",
    "DeepSeek-R1-Distill-Llama-70B": "#AEC7E8",
    "deepseek-r1-0528": "#1F3B87",
    "cogito-v1-preview-llama-70B": "#17BECF",
    "MiMo-7B-SFT": "#D62728",
    "Meditron3-70B": "#B2B2B2",
    "HuatuoGPT-o1-72B": "#E377C2",
    "o3-2025-04-16": "#FFDD57",
    "claude-sonnet-4": "#FFA07A",
    "gemini-2.5-pro": "#6A5ACD",
}


dtocamel = {
	'medbullets': 'MedBullets',
	'medexqa': 'MedExQA',
	'medmcqa': 'MedMCQA',
	'medqa': 'MedQA',
	'medxpertqa-r': 'MedXpert-R',
	'medxpertqa-u': 'MedXpert-U',
	'mmlu': 'MMLU',
	'mmlu-pro': 'MMLU-Pro',
	'pubmedqa': 'PubMedQA',
	'medarc': 'MedARC-QA',
	'metamedqa': 'MetaMedQA',
}


def main() -> None:
    records = []
    for dataset in datasets:
        for _ , models in id2llm.items():
            for model in models:
                model_trim = trim_first_segment(model)
                meld_full_path = f"{MELD_PATH}/{dataset}/{model_trim}/meld_results.json"
                with open(meld_full_path, 'r') as f:
                    data = json.load(f)
                    for similar_value in data["all_scores"]:
                        records.append({
                            "Dataset": dtocamel[data["dataset"]],
                            "Model": data["model"].split("/")[-1],
                            "Similarity": similar_value,
                        })

    df = pd.DataFrame(records)
    datasets_unique = df['Dataset'].unique()

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=True)
    axes = axes.flatten()

    # Hide any unused axes (because you have 11 datasets but 12 subplots)
    for ax in axes[len(datasets_unique):]:
        ax.set_visible(False)

    for ax, dataset in zip(axes, datasets_unique):
        subset = df[df['Dataset'] == dataset]
        subset = subset[subset['Model'].isin(model_order)]
        subset['Model'] = pd.Categorical(subset['Model'], categories=model_order, ordered=True)
        subset = subset.sort_values('Model')

        # Convert similarity to percentage
        subset['Similarity'] = subset['Similarity'] * 100

        sns.boxplot(data=subset, x='Model', y='Similarity', hue='Model', palette=palette, legend=False, ax=ax)
        ax.set_title(dataset)
        ax.set_ylabel('Similarity (%)')
        ax.set_xticklabels([])  # Hide x-axis tick labels
        ax.set_xlabel('')
        ax.set_ylim(0, 100)

    # Create legend handles for models (in order)
    handles = [mpatches.Patch(color=palette[m], label=m) for m in model_order if m in palette]
    # Create legend above plots, further up
    fig.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1))

    # Adjust layout to avoid overlap with legend
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(f'{MELD_PATH}/data_contamination_boxplot_grid.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
