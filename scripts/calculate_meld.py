import os
import re
import sys
import json
import argparse
import Levenshtein

from pathlib import Path
from datasets import Dataset

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from src.utils.preprocessing import load_benchmarks, trim_first_segment
from tiktoken import encoding_for_model
from src.utils.model_loading import initialize_llm_if_needed
from src.llm.llm.generating import llm_generation
from src.llm.gpt.generating import gpt_generation
from src.llm.open_router.generating import or_generation
from src.llm.gemini.generating import gemini_generation
from src.llm.llm.llm import Llm



DATA_PATH = "../data"


id2llm = {
    "llama": [
        # "meta-llama/Llama-3.2-1B-Instruct",
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
        # "open_router/google/gemini-2.5-pro",
        "open_router/deepseek/deepseek-r1-0528",
        "open_router/anthropic/claude-sonnet-4",
        "open_router/meta-llama/llama-4-maverick"
    ],
    "gemini": [
        "gemini/gemini-2.5-pro"
    ],
}


# instructions = """# Instructions:
# You are provided with the beginning of a known text enclosed within <known_text> tags.
# 
# If you recognize the source, continue it exactly as it appears in the original, maintaining wording, punctuation, capitalization, and formatting. Do not paraphrase or modify the content in any way.
# 
# Task:
# Complete the excerpt below by filling in the missing portion:
# 
# <known_text>"""

instructions = """# Instructions:
You will receive a passage enclosed within <text_to_continue>...</text_to_continue> tags. This passage contains the initial segment of a medical question.

Carefully analyze the passage to determine the most probable medical question being asked. Then, generate a clear, concise, and clinically appropriate continuation that completes the second half of the question.

Enclose your completion within <continue>...</continue> tags, maintaining a style consistent with the original text.
    
# Task:
<text_to_continue>"""

sys_prompt = f"""You are a helpful assistant who execute instructions.\n\n"""


def extract_last_known_text_block(text: str) -> str | None:
    pattern = re.compile(r"<continue>(.*?)</continue>", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else text


def construct_system_prompt() -> str:
    return [
        {
            "role": "system", 
            "content": sys_prompt
        }
    ]


def construct_user_prompt(
    prompt: str,
    llm_name: str
) -> str:
    if "gemma2" in llm_name.lower():
        content = f"""{sys_prompt}{instructions}{prompt}</text_to_continue>"""
    else:
        content = f"""{instructions}{prompt}</text_to_continue>"""
    return [
        {
            "role": "user", 
            "content": content
        }
    ]


def construct_prompt(
    prompt: str,
    llm_name: str
) -> str:
    if "gemma2" in llm_name.lower():
        return construct_user_prompt(
                prompt=prompt,
                llm_name=llm_name
            )
    else:
        return construct_system_prompt(
        ) + construct_user_prompt(
            prompt=prompt,
            llm_name=llm_name
        )
    

def truncate_text_by_token_length(text: str, max_tokens: int, tokenizer) -> str:
    token_ids = tokenizer.encode(text)
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
    truncated_text = tokenizer.decode(token_ids)
    return truncated_text


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_leven_ratio(
    expected_generation: list,
    extracted_generations: list,
    expected_len: list,
    tokenizer,
    threshold=0.95
) -> float:
    treshold_scores = []
    all_scores = []
    for exp_gen, ext_gen, exp_len in zip(expected_generation, extracted_generations, expected_len):
        reduced_text = truncate_text_by_token_length(ext_gen, exp_len, tokenizer)
        ratio = Levenshtein.ratio(exp_gen, reduced_text)
        all_scores.append(round(ratio, 2))
        if ratio > threshold:
            treshold_scores.append(round(ratio, 2))
    return all_scores, treshold_scores


def get_token_length(text: str, tokenizer) -> int:
    try:
        tokens = tokenizer.encode(text)
    except:
        tokens = tokenizer(text)
    return len(tokens)


def calculate_meld(
    llm: Llm | str,
    dataset: Dataset
) -> float:
    prompts = []
    expected_generations = []
    expected_len = []

    if isinstance(llm, Llm):
        tokenizer = llm.tokenizer
    else:
        tokenizer = encoding_for_model("gpt-4o")

    for sample in dataset:
        mid = len(sample["question"]) // 2
        prompt = sample["question"][:mid]
        prompts.append(prompt)
        expected_generations.append(sample["question"][mid:])
        expected_len.append(get_token_length(sample["question"][mid:], tokenizer))

    gen_params = {
        "temperature": 0.0,
        "seed": 0,
        "top_p": 0.9,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }

    dataset = dataset.add_column(
        "input_prompts",
        [
            construct_prompt(
                prompt=prompt,
                llm_name=llm if isinstance(llm, str) else llm.model_name
            )
            for prompt in prompts
        ]
    )

    if isinstance(llm, str):
        if "gpt" in llm or "o3" in llm:
            generated_texts = gpt_generation(llm=llm, gen_params=gen_params, dataset=dataset)
        elif "gemini" in llm:
            generated_texts = gemini_generation(llm=llm, gen_params=gen_params, dataset=dataset)
        else:
            generated_texts = or_generation(llm=llm, gen_params=gen_params, dataset=dataset)
    else:
        generated_texts = llm_generation(llm=llm, gen_params=gen_params, dataset=dataset)
    extracted_generations = [extract_last_known_text_block(generated_text) for generated_text in generated_texts]
    all_scores, threshold_scores = compute_leven_ratio(expected_generations, extracted_generations, expected_len, tokenizer)
    levan_tresh_score = (len(threshold_scores) / len(expected_generations)) * 100
    return all_scores, levan_tresh_score



def main(args: argparse.Namespace) -> None:
    benchmarks = load_benchmarks(benchmark_names=args.benchmarks)
    last_llm_name = None
    llm = None

    for llm_suite in args.llms:
        for llm_name in id2llm[llm_suite]:
            try:
                for benchmark in args.benchmarks:
                    print(f"LLM suite: {llm_suite} | LLM name: {trim_first_segment(llm_name)} | Benchmark: {benchmark}")
                    benchmark_path = Path(os.path.join(DATA_PATH, "benchmarks", benchmark))

                    dataset_names = [None]
                    if benchmark == "medagentsbench":
                        dataset_names = [d.name.split(".")[0] for d in benchmark_path.iterdir()]

                    for dataset_name in dataset_names:
                        llm, last_llm_name = initialize_llm_if_needed(llm_name, last_llm_name, llm)
                        if dataset_name:
                            dataset = benchmarks[benchmark][dataset_name]
                        else:
                            dataset = benchmarks[benchmark]

                        meld_path = Path(DATA_PATH) / "meld" / (dataset_name if dataset_name else benchmark) / trim_first_segment(llm_name) / "meld_results.json"
                        meld_path.parent.mkdir(parents=True, exist_ok=True)
                        if not meld_path.exists():
                            all_scores, meld_score = calculate_meld(llm=llm, dataset=dataset)
                            result = {
                                "dataset": dataset_name if dataset_name else benchmark,
                                "model": trim_first_segment(llm_name),
                                "all_scores": all_scores,
                                "meld_score": meld_score
                            }
                            
                            print(f"Dataset: {dataset_name if dataset_name else benchmark} | Model: {trim_first_segment(llm_name)} | MELD score: {meld_score}")
                            
                            os.makedirs(path, exist_ok=True)
                            with open(meld_path, "w") as f:
                                f.write(json.dumps(result) + "\n")
                        else:
                            print(f"Found MELD - Dataset: {dataset_name if dataset_name else benchmark} | Model: {trim_first_segment(llm_name)}")

                            
            except Exception as e:
                print(f"Error in loading model [{trim_first_segment(llm_name)}]: {e}")
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
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activate thinking mode (True or False)"
    )
    args = parser.parse_args()
    main(args)