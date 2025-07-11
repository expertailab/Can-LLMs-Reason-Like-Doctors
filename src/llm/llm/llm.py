from src.llm.llm import register_mimo_in_vllm

from vllm import LLM
from typing import Callable
from transformers import AutoTokenizer


class Llm():
    def __init__(
        self,
        model_name: str,
        dtype: str,
        seed: int
    ) -> Callable:
        self.tensor_parallel_size = 2 if "phi4" in model_name.split("/")[1].lower().replace("-", "") else 4
        self.dtype = "bfloat16" if "gemma-3" in model_name.split("/")[1].lower() else dtype
        self.model_prefix = model_name.split("/")[0]
        self.model_name = model_name.split("/")[1]
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{self.model_prefix}/{self.model_name}",
            trust_remote_code=True
        )
        self.model = LLM(
            model=f"{self.model_prefix}/{self.model_name}",
            dtype=self.dtype,
            seed=self.seed,
            max_model_len=4096,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True
        )