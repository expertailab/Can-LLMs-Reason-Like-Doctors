import re


def extract_latest_bracket_content(text: str) -> str | None:
    matches = re.findall(r'\[(.*?)\]', text)
    if matches:
        return matches[-1]
    return None


def extract_latest_angle_content(text: str) -> str | None: 
    matches = re.findall(r'\<(.*?)\>', text)
    if matches:
        return matches[-1]
    return None


def validate_output(
    data: dict,
    prompting_strategy: str
) -> list:
    clean_data = []
    for generated_solution in data:
        solution = extract_latest_bracket_content(generated_solution)
        if prompting_strategy in ["multi_init", "multi_debate"]:
            reasoning = extract_latest_angle_content(generated_solution)
            solution = f"Thinking: {reasoning}\nSolution: {solution}"
        clean_data.append(solution)
    return clean_data