import sys

from tqdm import tqdm
from openai import OpenAI

from src.utils.preprocessing import trim_first_segment


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

def or_generation(
    llm: str,
    dataset: list,
    gen_params: dict
) -> list:
    texts = []
    for sample in tqdm(dataset):
        try:
            response = client.chat.completions.create(
                model=trim_first_segment(llm),
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
    return texts