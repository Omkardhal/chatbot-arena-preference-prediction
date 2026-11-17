# Predicting Human Preference in Chatbot Arena

Kaggle competition: [LLM Classification Finetuning](https://www.kaggle.com/competitions/llm-classification-finetuning)

**Author:** Omkar (URN 2581394)
**Faculty:** Dr. Abhishek Pandey, Assistant Professor

## Task

Chatbot Arena users chat with two anonymous AI chatbots and vote for the answer they prefer.
Given a prompt and two responses, predict the probability that **response A wins**, **response B wins**, or the vote is a **tie**.
The metric is log loss (lower is better).

## Kaggle notebook

**Notebook:** [PASTE YOUR KAGGLE NOTEBOOK LINK HERE]

## Results

| Model | Log loss | Measured on |
|---|---|---|
| Guess 1/3 for each outcome | 1.0986 | reference |
| LightGBM baseline | 1.0235 | 5-fold cross-validation |
| **LightGBM baseline** | **1.02092** | **Kaggle public leaderboard** |
| LoRA fine-tune of Qwen2.5-1.5B (attempt, 2,500 rows) | 1.12 | own validation split, not submitted |

## Approach 1: text features + LightGBM (`baseline_lgbm.py`)

- Features per response: length, lines, bullets, code blocks, headers, bold text, digits, question marks, refusal phrases; computed for A, for B and as the difference A − B.
- Every training row is also added with A and B swapped (labels flipped); predictions average both views to cancel position bias.
- LightGBM with 5-fold stratified cross-validation and early stopping.

## Approach 2: LoRA fine-tune of a language model (`llm_pref_lora.py`)

- Qwen2.5-1.5B-Instruct with a 3-class head, LoRA adapters (rank 16) on 4-bit weights, so it fits a free Kaggle T4 GPU.
- Trained on a 2,500-row subset; validation log loss 1.12, worse than guessing, so it was not submitted.
- Likely cause: a randomly initialised classification head produced huge, overconfident outputs (training loss stayed near 9).
- A fix (zero-initialising the head) is in the script, but **the full rerun has not been completed**, so there is no score for it.

## How to run

On Kaggle, add the competition data to a notebook (internet off) and run `baseline_lgbm.py`; it writes `submission.csv`.
For the LoRA model, see the usage notes at the top of `llm_pref_lora.py` (needs a GPU).

## Next steps

Rerun the LoRA model with the zero-initialised head, train a larger model on the full data, blend LoRA and LightGBM predictions, and add text features such as TF-IDF.
