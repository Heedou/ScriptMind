# ScriptMind

---
# 📂 Input Data Format

The input file should be named **`task_SA_{train, dev, test}.json`**,  
and must follow the structure shown below.

Each entry represents one conversation sample with its metadata.

```json
{
  "conv_6133": {
  "conversation": "So, you have no knowledge about this at all?\nAlright, I understand.\nHave you ever visited the [address] branch yourself?\nThis is the Seoul Central District Prosecutors’ Office.\nOkay. So, you haven’t been there personally, right?\nYes. The bank account we found was opened around August 2015 at the [address] branch.\nThat’s why I asked you earlier.\nIn the past three years, have you ever lost your wallet, ID, or any personal items that could have led to a data leak?\nAccording to our cross-check with the financial institution, this account is indeed registered under your name.",
  "output": {
    "label": "vishing",
    "next_utterance": "The scammer’s next expected statement would be something like: We’re contacting you from the prosecution office to determine whether you personally opened this account and transferred it for money, or if your identity was stolen and you’re a victim of impersonation.",
    "rationale": "The scammer is currently trying to confirm whether the victim’s personal information was stolen and will likely proceed to question whether the victim sold the account or was impersonated, aiming to establish a false investigative context."
  },
  "vishing": "vishing",
  "source": "voice_phishing"
}
}
```

The dataset used in this research is available via the link below.

- **Dataset Link:** [🔗 Download ](https://drive.google.com/file/d/1WGJId_BOp9b0q3-VYn5etlTP9gXu1lox/view?usp=drive_link)

> ⚠️ **Note:** This dataset is provided for research and non-commercial use only.  
> Please cite the original authors and this project when using the dataset.

---
# 🚀 Running the Evaluation on Proprietary Models

After setting up your environment and API keys,  
you can run the evaluation script by specifying which model to use.

The command-line argument `--model` accepts one of the following options:
- `gpt` → OpenAI GPT (chatgpt-4o-latest)
- `gemini` → Google Gemini (gemini-2.0-flash-001)
- `claude` → Anthropic Claude (claude-3-5-haiku-20241022)

### 🧠 OpenAI GPT

Run evaluation using **ChatGPT (4o-latest)**:

```bash
python Evaluate_ProprietaryLLMs.py --model gpt
```

---

# 🧩 LLM Output Refinement & Extraction

This repository provides a unified post-processing pipeline for refining and extracting structured results from **Large Language Model (LLM)** outputs.  
It supports both **commercial models** (OpenAI GPT, Gemini, Claude) and **open-source models** (LLaMA, SOLAR, Exaone, etc.).

### 📘 Overview

When LLMs generate conversational classification outputs (e.g., voice phishing detection),  
their raw responses may include markdown formatting, invalid JSON, or unstructured text.  
This script cleans and standardizes those outputs, producing a consistent format suitable for evaluation.

### ✅ Key Features
- Works with **any LLM output CSV**
- Cleans code blocks (```json, ```python, etc.)
- Extracts structured fields from JSON or plain text
- Supports both **valid JSON** and **malformed outputs**
- Outputs unified columns for ground-truth and predictions

### 📂 Input Format

The script accepts CSV files that contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| **conversation** | string | The dialogue text analyzed by the LLM. |
| **output** | JSON / dict | Ground-truth structured label (if available). |
| **output_text** | string | Raw output text from the LLM (may include markdown or invalid JSON). |

### 💡 Example Input (`task_SA_gpt_results.csv`)

```csv
conversation,output,output_text
"Hello, this is the National Tax Service...", "{\"label\": \"vishing\"}", "```json\n{\"label\": \"vishing\", \"next_utterance\": \"Please confirm your account.\", \"rationale\": \"Scammer impersonating government.\"}\n```
```
### 🚀 Running the Script

Run the following command in your terminal:
```bash
python refine_llm_outputs.py
```

---
# 🤖 LLM-Based Automatic Evaluation for Crime Script Predictions

This repository provides a unified evaluation framework that automatically scores  
**LLM-generated predictions** for both the **next utterance** and **rationale** fields  
in scam detection datasets.

It uses an **LLM-as-a-judge** approach: GPT-4o-mini evaluates how semantically similar  
the model’s generated text is to the expert-annotated ground truth.


### 📘 Overview

When testing LLMs for cognitive or situational reasoning (e.g., in voice phishing scenarios),  
human evaluation is often expensive and inconsistent.  
This script automates that process by asking another LLM (GPT-4o-mini) to rate prediction quality.

Each output is scored from **0.00 to 1.00**, where:
- **1.00** → Perfect semantic match (identical meaning and intent)  
- **0.00** → Completely different or unrelated  
- **Intermediate (0.XX)** → Partial semantic overlap  

### 📂 Input Format

The script expects a preprocessed CSV file (from previous extraction steps)
located in the folder SA_experiments/, named as:

```bash
SA_experiments/{file}_extracted.csv
```

### 💡 Example Input (task_SA_gpt_test_results_extracted.csv)

```csv
conversation,gt_next_utterance,pred_next_utterance,gt_rationale,pred_rationale
"Victim: Hello? / Scammer: This is the National Tax Service.","We’re contacting you to verify a suspicious account.","We need to confirm your bank account.","The scammer impersonates a government officer to gain trust.","The scammer pretends to be an official to deceive the victim."
```

### 🚀 Running the Evaluation

Run the script by specifying the file name (without _extracted.csv):

```bash
python llm_auto_eval.py --file task_SA_gpt_test_results
```


