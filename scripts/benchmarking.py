import os
import sys
import string
import argparse

from pathlib import Path
from difflib import SequenceMatcher
from datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from src.llm.llm.llm import Llm
from src.utils.json_funcs import append_to_jsonl
from src.utils.preprocessing import load_benchmarks, trim_first_segment
from src.utils.model_loading import initialize_llm_if_needed
from src.llm_pass import (
    provide_single_solution,
    provide_multi_solution
)


DATA_PATH = "../data"


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
        # "open_router/deepseek/deepseek-r1-0528",
        "open_router/google/gemini-2.5-pro",
        # "open_router/anthropic/claude-sonnet-4",
        # "open_router/meta-llama/llama-4-maverick"
    ],
    "gemini": [
        "gemini/gemini-2.5-pro"
    ],
}


def process_options(example: dict) -> dict:
    if 'options' in example:
        if isinstance(example['options'], dict):
            example['options'] = list(example['options'].values())
    return example


def similar(data_sample: dict, clean_output: str) -> bool:
    best_sim_score = 0
    best_sim_string = ""
    best_index = ""
    index = 0

    options_field = data_sample.get("options")

    if isinstance(options_field, dict):
        options = [v for v in options_field.values() if v is not None]
    elif isinstance(options_field, list):
        options = [v for v in options_field if v is not None]
    else:
        options = []

    for option in options:
        sim = SequenceMatcher(None, clean_output, option).ratio()
        if sim > best_sim_score:
            best_sim_score = sim
            best_sim_string = option
            best_index = string.ascii_uppercase[index]
        index += 1
    if SequenceMatcher(None, clean_output, best_sim_string).ratio() >= 0.9:
        if best_index == data_sample["answer_idx"]:
            return True


def log_summary(
        action: str,
        count: int,
        llm_name: str,
        option: str,
        dataset_name: str,
        answer_type: str,
        prompt: str,
        seed: str
    ) -> None:
    print(
        f"{action} {count} solutions | "
        f"LLM: {llm_name} | "
        f"Options: {option} | "
        f"Dataset: {dataset_name} | "
        f"Answer type: {answer_type} | "
        f"Prompting strategy: {prompt} | "
        f"Seed: {seed}"
    )
    print('-' * 135)


def label_and_save_solutions(
        dataset: Dataset,
        option: str,
        output_path: str
    ) -> None:
    for sample in dataset:
        label = {"label": "different"}

        if sample["provided_solution"] is not None:
            if option == "free":
                if similar(sample, sample["provided_solution"]):
                    label = {"label": "similar"}

            elif option == "letter" and len(sample["provided_solution"]) < 15:
                if sample["gold_solution"] in sample["provided_solution"]:
                    label = {"label": "similar"}

        sample["similarity"] = label
        
        append_to_jsonl(
            {
                "id": sample["id"],
                "provided_solution": sample["provided_solution"],
                "gold_solution": sample["gold_solution"],
                "similarity": sample["similarity"]
            }, 
            output_path
        )


def get_predictions(
        predictions_jsonl: str,
        answer_type: str,
        prompt: str,
        dataset: Dataset,
        llm: Llm | str,
        args: argparse.Namespace,
        option: str,
        seed: str
    ) -> None:

    if option == "free":
        dataset = dataset.map(process_options)

    gen_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": seed,
        "enable_thinking": args.enable_thinking,
    }

    if prompt == "multi_persona":
        dataset = provide_multi_solution(
            llm=llm,
            gen_params=gen_params,
            dataset=dataset,
            prompting_strategy=prompt,
            answer_type=answer_type
        )
    else:
        dataset = provide_single_solution(
            llm=llm,
            gen_params=gen_params,
            dataset=dataset,
            prompting_strategy=prompt,
            answer_type=answer_type
        )

    dataset = dataset.add_column(
        "gold_solution",
        [sample["solution_idx"] if option == "letters" else sample["answer"] for sample in dataset]
    )

    llm_str = llm if isinstance(llm, str) else llm.model_name

    log_summary("Provided", len(dataset), llm_str, option, dataset.info.dataset_name, answer_type, prompt, seed)

    label_and_save_solutions(dataset, option, predictions_jsonl)

    log_summary("Labelled", len(dataset), llm_str, option, dataset.info.dataset_name, answer_type, prompt, seed)



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_predictions_path(
        base_path: str,
        benchmark: str,
        llm_name: str,
        answer_type: str,
        option: str,
        seed: int,
        prompt: str,
        enable_thinking: bool,
        dataset_name: str = None
    ) -> str:
    parts = [base_path, "predictions", benchmark]
    if dataset_name:
        parts.append(dataset_name)
    parts.extend([
        trim_first_segment(llm_name),
        f"answer_{answer_type}", f"option_{option}", f"seed_{seed}"
    ])
    predictions_dir = os.path.join(*parts)
    ensure_dir(predictions_dir)
    llm_id = trim_first_segment(llm_name).lower()
    if enable_thinking and ("cogito" in llm_id or "qwen3" in llm_id):
        filename = f"{prompt}_thinking.jsonl"
    else:
        filename = f"{prompt}.jsonl"
    return os.path.join(predictions_dir, filename)


def main(args: argparse.Namespace) -> None:
    benchmarks = load_benchmarks(benchmark_names=args.benchmarks)
    last_llm_name = None
    llm = None

    for llm_suite in args.llms:
        for llm_name in id2llm[llm_suite]:
            try:
                for benchmark in args.benchmarks:
                    print(f"LLM suite: {llm_suite} | LLM name: {trim_first_segment(llm_name)} | Benchmark: {benchmark}")
                    for answer_type in args.answer_types:
                        for option in args.options:
                            for seed in args.seeds:
                                for prompt in args.prompts:
                                    benchmark_path = Path(os.path.join(DATA_PATH, "benchmarks", benchmark))

                                    dataset_names = [None]
                                    if benchmark == "medagentsbench":
                                        dataset_names = [d.name.split(".")[0] for d in benchmark_path.iterdir()]

                                    for dataset_name in dataset_names:
                                        predictions_jsonl = get_predictions_path(
                                            base_path=DATA_PATH,
                                            benchmark=benchmark,
                                            llm_name=llm_name,
                                            answer_type=answer_type,
                                            option=option,
                                            seed=seed,
                                            prompt=prompt,
                                            enable_thinking=args.enable_thinking,
                                            dataset_name=dataset_name,
                                        )
                    
                                        if not os.path.exists(predictions_jsonl):
                                            llm, last_llm_name = initialize_llm_if_needed(llm_name, last_llm_name, llm)
                                            if dataset_name:
                                                dataset = benchmarks[benchmark][dataset_name]
                                            else:
                                                dataset = benchmarks[benchmark]

                                            get_predictions(
                                                predictions_jsonl=predictions_jsonl,
                                                answer_type=answer_type,
                                                prompt=prompt,
                                                dataset=dataset,
                                                llm=llm,
                                                args=args,
                                                option=option,
                                                seed=seed
                                            )
            except Exception as e:
                print(f"Error during *{trim_first_segment(llm_name)}* generation: {e}")
                continue




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
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling cutoff probability."
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activate thinking mode (True or False)"
    )
    args = parser.parse_args()
    main(args)