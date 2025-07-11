import time
import google.generativeai as genai

from tqdm import tqdm

from src.utils.preprocessing import trim_first_segment


genai.configure(api_key="")


def gemini_generation(
    llm: str,
    dataset: list,
    gen_params=dict
) -> list:
    generation_config = genai.GenerationConfig(
        max_output_tokens=gen_params["max_tokens"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
    )
    texts = []
    for sample in tqdm(dataset):
        try:
            gemini_connector = genai.GenerativeModel(
                model_name=trim_first_segment(llm),
                system_instruction=[prompt["content"] for prompt in sample["input_prompts"] if prompt["role"] == "system"][0]
            )
            generation = gemini_connector.generate_content(
                [prompt["content"] for prompt in sample["input_prompts"] if prompt["role"] == "user"][0],
                generation_config=generation_config
            )
            text = generation.text if generation.text else ""
        except Exception as e:
            print(e)
            text = ""
        texts.append(text)
        # time.sleep(5)
    return texts