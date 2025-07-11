import sys

from tqdm import tqdm
from openai import OpenAI


client = OpenAI(api_key="")


def gpt_generation(
    llm: str,
    dataset: list,
    gen_params: dict
) -> list:
    texts = []
    for sample in tqdm(dataset):
        try:
            if "o3" in llm.split('/')[1]:
                response = client.chat.completions.create(
                    model=llm.split('/')[1],
                    messages=sample["input_prompts"],
                    max_completion_tokens=gen_params["max_tokens"],
                )
            else:
                response = client.chat.completions.create(
                    model=llm.split('/')[1],
                    messages=sample["input_prompts"],
                    max_tokens=gen_params["max_tokens"],
                    temperature=gen_params["temperature"]
                )
            text = response.choices[0].message.content
        except Exception as e:
            print("ERROR: ", sys.exc_info()[0], e)
            text = None
        texts.append(text)
        # time.sleep(5)
    # print(texts)
    return texts