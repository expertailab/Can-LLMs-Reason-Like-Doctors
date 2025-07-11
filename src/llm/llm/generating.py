from vllm import (
    LLM,
    SamplingParams
)
from datasets import Dataset
from transformers import AutoTokenizer
from src.llm.llm.llm import Llm


def preprocess(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    gen_params: dict
):
    return {
        "input_prompts": tokenizer.apply_chat_template(
            dataset["input_prompts"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=gen_params["enable_thinking"]
        )
    }


def prompt_llm(
    llm: LLM,
    gen_params: dict,
    input_prompts: list
):
    params = SamplingParams(
        max_tokens=gen_params["max_tokens"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        seed=gen_params["seed"]
    )
    outputs = llm.generate(
        input_prompts,
        params
    )
    return outputs


def llm_generation(
    llm: Llm,
    gen_params: dict,
    dataset: Dataset,
):
    dataset = dataset.map(
        preprocess,
        fn_kwargs={
            "tokenizer": llm.tokenizer,
            "gen_params": gen_params
        }
    )
    # print(dataset[0]["input_prompts"])
    outputs = prompt_llm(
        llm=llm.model,
        gen_params=gen_params,
        input_prompts=dataset["input_prompts"],
    )
    # print(outputs[0].outputs[0].text)
    # print(X)
    return [
        output.outputs[0].text 
        for output in outputs
    ]