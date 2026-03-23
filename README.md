<h1>
  Can LLMs Reason Like Doctors?<br>
  <sub>Exploring the Limits of Large Language Models in Complex Medical Reasoning</sub>
</h1>

This repository hosts materials for the paper *"Can LLMs Reason Like Doctors? Exploring the Limits of Large Language Models in Complex Medical Reasoning"*, accepted at the *Findings* of [EACL 2026](https://2026.eacl.org/).

The work evaluates the parametric capabilities of LLMs in the context of complex medical reasoning and decision-making, assessing 77 LLMs with diverse fine-tuning approaches, ranging from 1B parameters to frontier models. Guided by medical problem-solving theory, three medical QA benchmarks have been selected to assess key abilities: reasoning processes (MedAgentsBench), susceptibility to cognitive biases (MedARC-QA), and metacognitive abilities (MetaMedQA). 

Additionally, a fine-grained dataset has been created by manually annotating a subset of questions to assess the *abduction*, *deduction*, and *induction* capabilities of LLMs, providing detailed insight into the reasoning mechanisms employed by physicians.

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-ffcc00?style=flat-square)](https://huggingface.co/datasets/expertailab/fine-grained-medical-reasoning)
[![ACL Paper](https://img.shields.io/badge/ACL%20Anthology-Paper-lightblue?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBmaWxsPSIjZWQyOTMzIiBkPSJNMTIyIDM0aDUzYzQuNSA3Ljg3NSA0LjUgNy44NzUgNC40ODggMTEuMDYybC4wMiAyLjE5Mi0uMDM4IDIuMzI4LjAwNyAyLjUxYy4wMDIgMi43MTctLjAyNCA1LjQzMy0uMDUxIDguMTVxLS4wMDcgMi44NC0uMDEgNS42OC0uMDEyIDUuOTMzLS4wNiAxMS44NjdhMjU3NiAyNTc2IDAgMCAwLS4wNjcgMTUuMjQ0Yy0uMDEgNC44NC0uMDMgOS42ODEtLjA1IDE0LjUyMmwtLjAxMSAyLjgxMnEtLjAxOSAzLjkxNC0uMDUxIDcuODI4bC0uMDA4IDIuMzY0Yy0uMDU0IDUuMzI2LS4wNTQgNS4zMjYtMS4xNjkgNi40NDFsMjIgMXY1NUg2di0ySDFxLS4wNy0yNS43MTMtLjEwNC01MS40MjUtLjAxNC0xMS45MzktLjA0Ny0yMy44NzctLjAzLTEwLjQwNC0uMDM3LTIwLjgxLS4wMDMtNS41MS0uMDIzLTExLjAyQy43NzIgNjkuNzcuNzcyIDY1LjY3Mi43NzMgNjEuNTcyTC43NSA1Ny44NzVsLjAwOC0zLjM4My0uMDA1LTIuOTM4QzEgNDkgMSA0OSAzIDQ1bDExOC0xem0tMiA2NHYySDU2bC0xIDI2LTIgMmg2OVY5OXoiLz48cGF0aCBmaWxsPSIjMDYwMTAxIiBkPSJtMiA0NiAyIDEtMSAxMjhoMTk3djhINnYtMkgxcS0uMDM1LTI1LjczOC0uMDUyLTUxLjQ3Ny0uMDA3LTExLjk1LS4wMjMtMjMuOVEuOTEgOTUuMjA4LjkwNSA4NC43OTVxMC01LjUxNy0uMDEtMTEuMDMyQy44ODUgNjkuNjYxLjg4NSA2NS41Ni44ODUgNjEuNDU4bC0uMDEtMy43MDguMDA0LTMuMzc4LS4wMDMtMi45NDFDMSA0OSAxIDQ5IDIgNDYiLz48cGF0aCBmaWxsPSIjMWYwNTA2IiBkPSJtNTIuODczIDg5Ljg2NCAyLjExLjAwNyAyLjIxMi0uMDA2cTMuNjM2LS4wMDQgNy4yNzQuMDAybDUuMDMzLS4wMDNxNS4yOC0uMDAxIDEwLjU2Mi4wMDYgNi43ODQuMDA2IDEzLjU3LS4wMDQgNS4xOTctLjAwNCAxMC4zOTYgMCAyLjUwMyAwIDUuMDA1LS4wMDIgMy40OTItLjAwMyA2Ljk4My4wMDdsMi4xMS0uMDA3YzQuNzU4LjAyMiA0Ljc1OC4wMjIgNS44NzIgMS4xMzYuMDg5IDIuODg0LjExNSA1Ljc0NS4wOTggOC42MjlsLS4wMDMgMi4zMmMtLjAwNiAyLjg3MS0uMDIgNS43NDItLjAzMyA4LjYxM0wxMjQgMTMwSDQ3bC0uMDYzLTE5LjQzOC0uMDI3LTYuMTQyLS4wMDgtNC43OTEtLjAxNS0yLjUzNmMwLTIuMDMyLjA1LTQuMDYzLjExMy02LjA5MyAxLjY1OS0xLjY1OSAzLjY1Ny0xLjEyNiA1Ljg3My0xLjEzNk0xMjAgOTh2Mkg1NmwtMSAyNi0yIDJoNjlWOTl6Ii8+PHBhdGggZmlsbD0iIzBiMDEwMiIgZD0iTTEyMyAzNGg1MmM0LjUgNy44NzUgNC41IDcuODc1IDQuNDg4IDExLjA2MmwuMDIgMi4xOTItLjAzOCAyLjMyOC4wMDcgMi41MWMuMDAyIDIuNzE3LS4wMjQgNS40MzMtLjA1MSA4LjE1cS0uMDA3IDIuODQtLjAxIDUuNjgtLjAxMiA1LjkzMy0uMDYgMTEuODY3YTI1NzYgMjU3NiAwIDAgMC0uMDY3IDE1LjI0NGMtLjAxIDQuODQtLjAzIDkuNjgxLS4wNSAxNC41MjJsLS4wMTEgMi44MTJxLS4wMTkgMy45MTQtLjA1MSA3LjgyOGwtLjAwOCAyLjM2NGMtLjA1NCA1LjMyNi0uMDU0IDUuMzI2LTEuMTY5IDYuNDQxbDIyIDF2MmgtMzBWMzZoLTQ3eiIvPjxwYXRoIGZpbGw9IiM3NDI2MmEiIGQ9Im0xMjIgMzQgMSAyaDQ3djk0aDMwdjFoLTMxVjM3aC00NmwxIDN2N0g0bC0xLTIgMTE4LTF6Ii8+PC9zdmc+)](https://aclanthology.org/YOUR_PAPER_ID/)



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

## Citation

Please cite the following work if you use this dataset or code: 

```
@inproceedings{merenda-etal-2026-llms,
    title = "Can {LLM}s Reason Like Doctors? Exploring the Limits of Large Language Models in Complex Medical Reasoning",
    author = "Merenda, Flavio  and
      Gomez-Perez, Jose Manuel  and
      Rigau, German",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.127/",
    pages = "2432--2452",
    ISBN = "979-8-89176-386-9",
    abstract = "Large language models (LLMs) have shown remarkable progress in reasoning across multiple domains. However, it remains unclear whether their abilities reflect genuine reasoning or sophisticated pattern matching, a distinction critical in medical decision-making, where reliable multi-step problem-solving is required. Accordingly, we conduct one of the largest evaluations to date, assessing 77 LLMs with diverse fine-tuning approaches, ranging from 1 billion parameters to frontier models. Guided by medical problem-solving theory, we select three medical question answering (QA) benchmarks targeting key reasoning skills: reasoning processes, susceptibility to cognitive biases, and metacognitive abilities. Additionally, we manually annotate a subset of questions to assess the abduction, deduction, and induction capabilities of LLMs, offering detailed insight into the reasoning mechanisms followed by physicians, an aspect that has received relatively limited attention in this domain. Most models, particularly smaller ones, struggle even with specialized fine-tuning or advanced prompting. Larger models perform better but still show clear limitations in complex medical reasoning. Our findings highlight the need to improve specific reasoning strategies to better reflect medical decision-making. The datasets and code used in this study are publicly available at: \url{https://github.com/expertailab/Can-LLMs-Reason-Like-Doctors}"
}
```

