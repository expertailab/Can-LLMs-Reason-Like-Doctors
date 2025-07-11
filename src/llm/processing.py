from datasets import Dataset
from src.utils.preprocessing import construct_input
from src.utils.data_validation import validate_output
from src.llm.gemini.generating import gemini_generation
from src.llm.gpt.generating import gpt_generation
from src.llm.llm.generating import llm_generation
from src.llm.open_router.generating import or_generation
from src.llm.llm.llm import Llm


def get_output(
    llm: Llm | str,
    gen_params: dict,
    dataset: Dataset,
    prompting_strategy: str = ""
) -> list:
    llm_str = llm if isinstance(llm, str) else llm.model_name
    if "gemini" in llm_str.lower():
        return validate_output(
            data=gemini_generation(
                llm=llm,
                dataset=dataset,
                gen_params=gen_params
            ),
            prompting_strategy=prompting_strategy
        )
    elif "gpt" in llm_str.lower():
        return validate_output(
            data=gpt_generation(
                llm=llm,
                dataset=dataset,
                gen_params=gen_params
            ),
            prompting_strategy=prompting_strategy
        )
    elif "open_router" in llm_str.lower():
        return validate_output(
            data=or_generation(
                llm=llm,
                dataset=dataset,
                gen_params=gen_params
            ),
            prompting_strategy=prompting_strategy
        )
    else:
        return validate_output(
            data=llm_generation(
                llm=llm,
                gen_params=gen_params,
                dataset=dataset
            ),
            prompting_strategy=prompting_strategy
        )


def process(
    llm: Llm | str,
    gen_params: dict,
    dataset: Dataset,
    prompting_strategy: str,
    answer_type: str,
    role: str = ""
) -> list:
    dataset = dataset.add_column(
        "input_prompts",
        construct_input(
            dataset=dataset,
            prompting_strategy=prompting_strategy,
            answer_type=answer_type,
            llm_name=llm if isinstance(llm, str) else llm.model_name,
            role=role
        )
    )
    return get_output(
        llm=llm,
        gen_params=gen_params,
        dataset=dataset,
        prompting_strategy=prompting_strategy
    )