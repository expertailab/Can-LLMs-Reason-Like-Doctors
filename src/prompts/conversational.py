from src.prompts.few_shots import get_few_shot


def construct_system_prompt(
    prompting_strategy: str,
    role: str
) -> str:
    if prompting_strategy == "base":
        system_prompt = f"""You are a helpful assistant who execute instructions."""
    elif prompting_strategy == "multi_init":
        system_prompt = f"""You are a {role}."""
    elif prompting_strategy == "multi_debate":
        system_prompt = f"""You are a {role}."""
    elif prompting_strategy == "multi_final":
        system_prompt = """You are a senior medical expert."""
    else:
        system_prompt = f"""You are a helpful assistant who execute instructions to solve clinical problems."""
    return [
        {
            "role": "system", 
            "content": system_prompt
        }
    ]


def open_question(data_sample: dict) -> str:
    return f"""# Clinical problem:\n{data_sample["question"]}"""


def closed_question(data_sample: dict, dataset_name: str) -> str:
    if isinstance(data_sample["options"], dict): 
        data_sample["options"] = {key: value for key, value in data_sample["options"].items() if value is not None}
    if isinstance(data_sample["options"], list):
        data_sample["options"] = [item for item in data_sample["options"] if item is not None]
    if dataset_name in ["medxpertqa-r", "medxpertqa-u"]:
        question = data_sample["question"].split("Answer Choices:")[0]
    else:
        question = data_sample["question"]
    return f"""# Clinical problem:\n{question}\n\n# Options:\n{data_sample["options"]}\n"""


def construct_question(
    dataset_name: str, 
    answer_type: str,
    data_sample: dict
) -> str:
    if answer_type == "open":
        return open_question(data_sample)
    if answer_type == "closed":
        return closed_question(data_sample, dataset_name)
        

FORMAT = """Select the most appropriate solution from the provided 'Options' list and report it within square brackets [ ]"""

REASONING = """Report your reasoning steps within angle brackets < >."""

BASE_INSTRUCTIONS = f"""{FORMAT}"""

COT_INSTRUCTIONS = f"""Reason step-by-step to determine the correct solution. {FORMAT}"""

IR_INSTRUCTIONS = f"""Use the available medical information to deduce and determine the correct solution. {FORMAT}""" 

AP_INSTRUCTIONS = f"""Retrieve three distinct medical problems, each different from one another and from the provided problem, followed by your step-by-step reasoning to determine the correct solution. {FORMAT}"""

ARR_INSTRUCTIONS = f"""Identify the goal of the problem, retrieve relevant information, and step-by-step reason to solve the problem. {FORMAT}"""

FEWSHOT_INSTRUCTIONS = f"""Select the most appropriate solution from the provided 'Options' list, furnish an explanation for your choice, and write the selected solution inside square brackets [ ]"""

MULTI_INIT_INSTRUCTIONS = f"""Based on your medical expertise, please analyze the provided problem and options. Think step-by-step and provide your reasoning and answer clearly. Please include distinct sections for your thinking and your answer in your response. {REASONING} {FORMAT}"""

MULTI_DEBATE_INSTRUCTIONS = f"""Please update your analysis for the question. Think carefully step-by-step and revise your answer accordingly. Provide your response with clear sections for your updated thinking and updated answer. {REASONING} {FORMAT}"""

MULTI_FINAL_INSTRUCTIONS = f"""Please carefully review all the information and provide the final decision. Offer a detailed rationale and clearly state your final answer, indicating your final reasoning and the chosen option. {REASONING} {FORMAT}"""


def select_instructions(prompting_strategy: str) -> str:
    if prompting_strategy == "base": return BASE_INSTRUCTIONS
    if prompting_strategy == "cot": return COT_INSTRUCTIONS
    if prompting_strategy == "ir": return IR_INSTRUCTIONS
    if prompting_strategy == "ap": return AP_INSTRUCTIONS
    if prompting_strategy == "arr": return ARR_INSTRUCTIONS
    if prompting_strategy == "few_shot": return FEWSHOT_INSTRUCTIONS
    if prompting_strategy == "multi_init": return MULTI_INIT_INSTRUCTIONS
    if prompting_strategy == "multi_debate": return MULTI_DEBATE_INSTRUCTIONS
    if prompting_strategy == "multi_final": return MULTI_FINAL_INSTRUCTIONS


debate_roles = [
    "Innovative Medical Thinker - MD",
    "Critical Medical Analyst - Medical Professor",
    "Clinical Decision Specialist - Medical Researcher"
]


def construct_user_prompt(
    dataset_name: str,
    answer_type: str,
    data_sample: dict,
    prompting_strategy: str,
    llm_name: str,
    role: str
) -> str:
    few_shots = []
    if prompting_strategy == "few_shot":
        few_shots = get_few_shot(llm_name)
    instructions = select_instructions(
        prompting_strategy=prompting_strategy
    )
    question = construct_question(
        dataset_name=dataset_name,
        answer_type=answer_type,
        data_sample=data_sample
    )
    context = ""
    if prompting_strategy == "base":
        system_prompt = f"""You are a helpful assistant who execute instructions."""
    elif prompting_strategy == "multi_init":
        system_prompt = f"""You are a {role}."""
    elif prompting_strategy == "multi_debate":
        context_turns = [f"- {rol}'s previous opinion:\n{data_sample[f'multi_init_{str(i)}']}" for i, rol in enumerate(debate_roles) if rol != role]
        joined_context = '\n'.join(context_turns)
        system_prompt = f"""You are a {role}."""
        context = f"""\nConsidering the following insights from your peers:\n{joined_context}"""
    elif prompting_strategy == "multi_final":
        context_turns = [f"- {rol}'s final opinion:\n{data_sample[f'multi_debate_{str(i)}']}" for i, rol in enumerate(debate_roles)]
        joined_context = '\n'.join(context_turns)
        system_prompt = """You are a senior medical expert."""
        context = f"""\nConsidering all the following debate insights and answers:\n{joined_context}"""
    else:
        system_prompt = f"""You are a helpful assistant who execute instructions to solve clinical problems."""
    if "gemma2" in llm_name.lower():
        content = f"""{system_prompt}\n\n{question}\n# Instructions:{context}\n{instructions}"""
    else:
        content = f"""{question}\n# Instructions:{context}\n{instructions}"""
    return few_shots + [
        {
            "role": "user", 
            "content": content
        }
    ]


def construct_prompt(
    dataset_name: str,
    answer_type: str,
    data_sample: dict,
    prompting_strategy: str,
    llm_name: str,
    role: str
) -> str:
    if "gemma2" in llm_name.lower():
        return construct_user_prompt(
                dataset_name=dataset_name,
                answer_type=answer_type,
                data_sample=data_sample,
                prompting_strategy=prompting_strategy,
                llm_name=llm_name,
                role=role
            )
    else:
        return construct_system_prompt(
            prompting_strategy=prompting_strategy,
            role=role
        ) + construct_user_prompt(
            dataset_name=dataset_name,
            answer_type=answer_type,
            data_sample=data_sample,
            prompting_strategy=prompting_strategy,
            llm_name=llm_name,
            role=role
        )