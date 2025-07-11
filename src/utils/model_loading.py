import os
import sys

from typing import Union
from pathlib import Path
from vllm.distributed import cleanup_dist_env_and_memory

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

from src.llm.llm.llm import Llm



def initialize_llm_if_needed(
        current_llm_name: str,
        last_llm_name: str,
        llm: Llm | str | None
    ) -> tuple:
    if current_llm_name != last_llm_name:
        if "gpt" in current_llm_name:
            return current_llm_name, current_llm_name
        if "o3" in current_llm_name:
            return current_llm_name, current_llm_name
        if "open_router" in current_llm_name:
            return current_llm_name, current_llm_name
        if "gemini" in current_llm_name:
            return current_llm_name, current_llm_name
        if llm:
            del llm.model
            cleanup_dist_env_and_memory()
        llm = Llm(model_name=current_llm_name, dtype="auto", seed=0)
        last_llm_name = current_llm_name
    return llm, last_llm_name