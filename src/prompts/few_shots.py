import random

from datasets import load_dataset



def get_few_shot(llm_name: str) -> list:
   if "gemma2" not in llm_name.lower():
      dataset = load_dataset("json", data_files="../data/few_shot/casimedicos.jsonl", split="train")
      subset = random.sample(list(dataset), 3)
      few_shots = []
      for sample in subset:
         sample["options"] = list(sample['options'].values())
         few_shots.append(
            {
               "role": "user",
               "content": f"""# Clinical problem:\n{sample["question"]}\n\n# Options:\n{sample["options"]}\n"""
            }
         )
         few_shots.append(
            {
               "role": "assistant", 
               "content": f"""# Explanation:\n{sample["full_answer"]}\n\n[{sample["answer"]}]\n"""
            }
         )
      return few_shots