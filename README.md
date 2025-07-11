# Not Yet Good Doctors: Exploring the Limits of LLMs in Complex Medical Reasoning

This repository contains materials for **"Not Yet Good Doctors: Exploring the Limits of LLMs in Complex Medical Reasoning"**, which evaluates the parametric capabilities of large language models (LLMs) in the context of complex medical reasoning and decision-making.

This study provides a systematic evaluation of LLMs across four model categories:

- **General-purpose models**
- **Reasoning-optimized models**
- **Medically specialized models**
- **Hybrid models with both reasoning and medical tuning**


## Repository Structure

```
data/
├── benchmarks/
├── few_shot/
├── meld/
└── predictions/
scripts/
src/
```

---

**Folder descriptions:**

- `benchmarks/` — Preprocessed benchmark datasets (MedAgentsBench, MedARC-QA, MetaMedQA)  
- `few_shot/` — Casimedicos dataset and processing script for few-shot experiments  
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

