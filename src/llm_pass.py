from datasets import Dataset
from src.llm.processing import process
from src.llm.llm.llm import Llm


def provide_single_solution(
    llm: Llm | str,
    gen_params: dict,
    dataset: Dataset,
    prompting_strategy: str,
    answer_type: str
) -> Dataset:
    dataset = dataset.add_column(
        "provided_solution",
        process(
            llm=llm,
            gen_params=gen_params,
            dataset=dataset, 
            prompting_strategy=prompting_strategy,
            answer_type=answer_type
        )
    )
    return dataset


def provide_multi_solution(
    llm: Llm,
    gen_params: dict,
    dataset: Dataset,
    prompting_strategy: str,
    answer_type: str
) -> Dataset:
    if prompting_strategy == "multi_persona":
        debate_roles = [
            "Innovative Medical Thinker - MD",
            "Critical Medical Analyst - Medical Professor",
            "Clinical Decision Specialist - Medical Researcher"
        ]
        for i, role in enumerate(debate_roles):
            dataset = dataset.add_column(
                f"multi_init_{i}",
                process(
                    llm=llm,
                    gen_params=gen_params,
                    dataset=dataset, 
                    prompting_strategy="multi_init",
                    answer_type=answer_type,
                    role=role
                )
            )
        for i, role in enumerate(debate_roles):
            dataset = dataset.add_column(
                f"multi_debate_{i}",
                process(
                    llm=llm,
                    gen_params=gen_params,
                    dataset=dataset, 
                    prompting_strategy="multi_debate",
                    answer_type=answer_type,
                    role=role
                )
            )
        dataset = dataset.add_column(
            "provided_solution",
            process(
                llm=llm,
                gen_params=gen_params,
                dataset=dataset, 
                prompting_strategy="multi_final",
                answer_type=answer_type
            )
        )
        return dataset
    else:
        raise ValueError(
            f"No agentic pipeline available for this prompting strategy: {prompting_strategy}"
        )