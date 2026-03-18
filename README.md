<h1>
  Can LLMs Reason Like Doctors?<br>
  <sub>Exploring the Limits of Large Language Models in Complex Medical Reasoning</sub>
</h1>

This repository hosts materials for the paper *"Can LLMs Reason Like Doctors? Exploring the Limits of Large Language Models in Complex Medical Reasoning"*, accepted at the *Findings* of [EACL 2026](https://2026.eacl.org/).

The work evaluates the parametric capabilities of LLMs in the context of complex medical reasoning and decision-making, assessing 77 LLMs with diverse fine-tuning approaches, ranging from 1B parameters to frontier models. Guided by medical problem-solving theory, three medical QA benchmarks have been selected to assess key abilities: reasoning processes (MedAgentsBench), susceptibility to cognitive biases (MedARC-QA), and metacognitive abilities (MetaMedQA). 

Additionally, a fine-grained dataset has been created by manually annotating a subset of questions to assess the *abduction*, *deduction*, and *induction* capabilities of LLMs, providing detailed insight into the reasoning mechanisms employed by physicians.

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc00?style=flat-square)](https://huggingface.co/datasets/expertailab/fine-grained-medical-reasoning)




## Repository Structure

```
data/
├── benchmarks/
├── few_shot/
├── fine_grained/
├── meld/
└── predictions/
scripts/
src/
```

---

**Folder descriptions:**

- `benchmarks/` — Preprocessed benchmark datasets (MedAgentsBench, MedARC-QA, MetaMedQA)  
- `few_shot/` — Casimedicos dataset and processing script for few-shot experiments 
- `fine_grained/` — Data annotated for fine-grained reasoning analysis
- `meld/` — Folder for storing result of contamination analysis  
- `predictions/` — Folder for storing model outputs from evaluation experiments  
- `src/` — Source code
- `scripts/` — Scripts to run experiments


## Setting Up the Conda Environment

Follow these steps to create a Conda environment and install the required Python packages from `requirements.txt`.

### Create and activate a new Conda environment

```bash
conda create -n nygd python=3.9 -y
conda activate nygd
```

### Install packages
```bash
pip install -r requirements.txt
```

## Running Experiments

### Calculating random baseline

```bash
python calculate_random_baseline.py
```

### Benchmarking

```bash
chmod +x run_benchmarking.sh
./run_benchmarking.sh
```

### Calculating data contamination

```bash
chmod +x run_meld.sh
./run_meld.sh
```

### Evaluating

```bash
chmod +x run_meld.sh
./run_meld.sh
```

## Plotting results

```bash
python plotting_meld.py
```

```bash
python plotting_prompts.py
```

```bash
python plotting_fine_grained_analysis.py
```

