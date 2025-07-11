import os
import sys
import argparse

from pathlib import Path

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

import numpy as np
import pandas as pd

from src.utils.preprocessing import load_benchmarks
from typing import Union
from datasets import (
    load_dataset,
    ClassLabel
)
from transformers import AutoTokenizer



DATA_PATH = "../data"


row_order = [
    'medagentsbench',
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


medagentsbench = [
	'medbullets',
	'medexqa',
	'medmcqa',
	'medqa',
	'medxpertqa-r',
	'medxpertqa-u',
	'mmlu',
	'mmlu-pro',
	'pubmedqa',
]


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
    

def main(args: argparse.ArgumentParser) -> None:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    benchmarks_info = []
    benchmarks = load_benchmarks(benchmark_names=args.benchmarks)
    for benchmark in args.benchmarks:
        benchmark_path = Path(os.path.join(DATA_PATH, "benchmarks", benchmark))

        dataset_names = [None]
        if benchmark == "medagentsbench":
            dataset_names = [d.name.split(".")[0] for d in benchmark_path.iterdir()]
        for dataset_name in dataset_names:
            if dataset_name:
                dataset = benchmarks[benchmark][dataset_name]
            else:
                dataset = benchmarks[benchmark]

            benchmark_info = {}

            min_options = min(len([opt for opt in item.get('options', []) if opt is not None]) for item in dataset)
            max_options = max(len([opt for opt in item.get('options', []) if opt is not None]) for item in dataset)

            if min_options == max_options:
                options = max_options
            else:
                options = f"{min_options}-{max_options}"

            benchmark_info["benchmark"] = dataset.info.dataset_name
            benchmark_info["size"] = len(dataset)
            benchmark_info["avg_token_length"] = int(sum(len(tokenizer.encode(example["question"], add_special_tokens=False)) for example in dataset) / len(dataset))
            benchmark_info["options"] = options
            benchmarks_info.append(benchmark_info)

    min_options = min(item['options'] for item in benchmarks_info if item["benchmark"] in medagentsbench)
    max_options = max(item['options'] for item in benchmarks_info if item["benchmark"] in medagentsbench)

    benchmarks_info.append(
        {
            "benchmark": "medagentsbench",
            "size": sum(item["size"] for item in benchmarks_info if item["benchmark"] in medagentsbench),
            "avg_token_length": int(sum(item["avg_token_length"] for item in benchmarks_info if item["benchmark"] in medagentsbench) / len(benchmarks_info)),
            "options": f"{min_options}-{max_options}"
        }
    )

    df = pd.DataFrame(benchmarks_info, columns=list(benchmarks_info[0].keys()))
    latex_df = df.set_index("benchmark").loc[row_order].reset_index()
    print(latex_df)
    print(latex_df.to_latex(header=True, index=False))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks", 
        nargs="+",
        choices=["medagentsbench", "medarc", "metamedqa"],
        default=["medagentsbench", "medarc", "metamedqa"],
        help="List of datasets to process."
    )
    args = parser.parse_args()
    main(args)
