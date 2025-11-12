"""
===========================================================
LLM-Based Automatic Evaluation for Voice Phishing Predictions
===========================================================

This script automatically evaluates LLM-generated predictions
for both 'next_utterance' and 'rationale' fields by comparing
them to expert-annotated ground truths using a scoring LLM.

Each comparison receives a semantic similarity score
between 0.00 and 1.00:
- 1.00 → Fully equivalent meaning (same phishing intent/context)
- 0.00 → Completely different meaning
- 0.XX → Partial semantic overlap

-----------------------------------------------------------
Usage:
    export OPENAI_API_KEY="your_openai_key"
    python llm_auto_eval.py --file task_SA_gpt_test_results

Input:
    SA_experiments/{file}_extracted.csv
Output:
    SA_experiments/{file}_scored.csv

-----------------------------------------------------------
"""

import os
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import argparse

# ===========================================================
# 1. Initialize OpenAI Client
# ===========================================================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===========================================================
# 2. Evaluation Prompts
# ===========================================================
NEXT_UTTERANCE_PROMPT = """You are an expert evaluator for voice phishing scenario predictions.
Your task is to compare the model's predicted next utterance with the correct ground truth utterance.
Rate the prediction STRICTLY based on whether it conveys the *same phishing situation* or meaning as the ground truth,
not merely based on text similarity.

Give a score between 0.00 and 1.00 (two decimal places):
- 1.00 → fully matches the meaning and intent of the ground truth
- 0.00 → completely different or unrelated
- intermediate values (e.g., 0.45, 0.72) → partial semantic overlap or situational similarity.

Output **only the numeric score**, no explanation."""

RATIONALE_PROMPT = """You are an expert evaluator for voice phishing scenario reasoning.
Your task is to compare the model’s predicted rationale (explanation of the scam’s intent)
with the correct expert-annotated rationale.

Rate how closely the model’s reasoning aligns with the ground truth, 
based on semantic and situational similarity, not literal wording.

Give a score between 0.00 and 1.00 (two decimal places):
- 1.00 → fully matches the meaning and reasoning intent of the ground truth
- 0.00 → completely different or irrelevant
- intermediate values → partial semantic alignment.

Output **only the numeric score**, no explanation."""

# ===========================================================
# 3. LLM-based Scoring Function
# ===========================================================
def get_llm_score(gt_text: str, pred_text: str, system_prompt: str) -> float:
    """
    Requests an evaluation score (0.00–1.00) from the LLM.
    """
    user_prompt = f"""Ground Truth:
{gt_text}

Model Prediction:
{pred_text}

Score:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        score_text = response.choices[0].message.content.strip()
        try:
            score = float(score_text)
            if 0 <= score <= 1:
                return round(score, 2)
        except ValueError:
            return None
    except Exception as e:
        print(f"⚠️ Error evaluating sample: {e}")
        return None


# ===========================================================
# 4. Evaluation Runner
# ===========================================================
def evaluate_file(file_name: str):
    """
    Evaluates both next_utterance and rationale similarity
    for all samples in the extracted CSV.
    """
    input_path = f"SA_experiments/{file_name}_extracted.csv"
    output_path = f"SA_experiments/{file_name}_scored.csv"

    df = pd.read_csv(input_path)
    df["llm_score_next_utterance"] = None
    df["llm_score_rationale"] = None

    print(f"🚀 Evaluating file: {input_path}")
    for i in tqdm(range(len(df)), desc="Evaluating samples"):
        gt_next = str(df.loc[i, "gt_next_utterance"])
        pred_next = str(df.loc[i, "pred_next_utterance"])
        gt_rat = str(df.loc[i, "gt_rationale"])
        pred_rat = str(df.loc[i, "pred_rationale"])

        # Evaluate next_utterance
        score_next = get_llm_score(gt_next, pred_next, NEXT_UTTERANCE_PROMPT)
        # Evaluate rationale
        score_rat = get_llm_score(gt_rat, pred_rat, RATIONALE_PROMPT)

        df.loc[i, "llm_score_next_utterance"] = score_next
        df.loc[i, "llm_score_rationale"] = score_rat

    df.to_csv(output_path, index=False)
    print(f"✅ Evaluation completed! Results saved to: {output_path}")


# ===========================================================
# 5. Main Entry Point
# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic LLM evaluation for next_utterance and rationale fields.")
    parser.add_argument("--file", required=True, help="Base name of the extracted CSV (without extension)")
    args = parser.parse_args()

    evaluate_file(args.file)
