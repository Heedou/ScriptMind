"""
===========================================================
LLM Output Refinement & Extraction Script
===========================================================

This script processes the raw CSV results from both commercial
and open-source Large Language Models (LLMs) for voice phishing detection.

It extracts structured fields (label, next_utterance, rationale)
from model outputs and ground-truth columns, producing clean
and standardized CSV files for analysis.

-----------------------------------------------------------
Usage:
    python refine_llm_outputs.py

Input:
    One or multiple CSV files (e.g., GPT, Gemini, Claude, LLaMA, Exaone...)

Output:
    *_extracted.csv files saved in the same directory
-----------------------------------------------------------
"""

import pandas as pd
import json
import re
import ast
from pathlib import Path


# ===========================================================
# 1. JSON Cleaner
# ===========================================================
def clean_json_text(text):
    """Removes markdown code blocks and trims whitespace."""
    if pd.isna(text):
        return None
    text = str(text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)  # remove ```json, ```python, etc.
    text = re.sub(r"```$", "", text.strip())      # remove ending ```
    return text.strip()


# ===========================================================
# 2. Generic JSON Parser
# ===========================================================
def parse_json_field(value, key):
    """Extracts a specific field from a JSON string."""
    try:
        if pd.isna(value) or not value.strip():
            return None
        cleaned_value = clean_json_text(value)
        data = json.loads(cleaned_value)
        return data.get(key)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


# ===========================================================
# 3. Open-source LLM Output Parser (Non-strict JSON)
# ===========================================================
def parse_loose_llm_output(text):
    """
    Handles unstructured model outputs (e.g., open-source LLMs)
    that may contain malformed or mixed content.
    """
    if not isinstance(text, str):
        return pd.Series(["", "", ""])

    parsed_part = text.split("NO EXTRA TEXT")[-1] if "NO EXTRA TEXT" in text else text

    # Label extraction
    label = ""
    if "non_vishing" in parsed_part:
        label = "non_vishing"
    elif "vishing" in parsed_part:
        label = "vishing"

    # Next utterance
    if "next_utterance" in parsed_part:
        next_part = parsed_part.split("next_utterance")[-1]
        next_utt = next_part.split("rationale")[0].strip()
    else:
        next_utt = ""

    # Rationale
    rationale = parsed_part.split("rationale")[-1] if "rationale" in parsed_part else ""

    return pd.Series([label, next_utt, rationale])


# ===========================================================
# 4. Ground Truth Parser
# ===========================================================
def parse_output_dict(text):
    """Parses the ground-truth JSON-like dictionary in 'output' column."""
    try:
        if isinstance(text, str):
            data = ast.literal_eval(text)
            label = data.get("label", "")
            next_utt = data.get("next_utterance", "")
            rationale = data.get("rationale", "")
            return pd.Series([label, next_utt, rationale])
        else:
            return pd.Series(["", "", ""])
    except Exception:
        return pd.Series(["", "", ""])


# ===========================================================
# 5. Main Processing Function
# ===========================================================
def process_csv(input_path: str):
    """Reads a CSV file, parses outputs, and saves an extracted version."""
    df = pd.read_csv(input_path, encoding="utf-8")

    # Parse ground-truth outputs
    df[["gt_label", "gt_next_utterance", "gt_rationale"]] = df["output"].apply(parse_output_dict)

    # Try JSON-based parsing first (for GPT/Gemini/Claude)
    df["pred_label_json"] = df["output_text"].apply(lambda x: parse_json_field(x, "label"))
    df["pred_next_json"] = df["output_text"].apply(lambda x: parse_json_field(x, "next_utterance"))
    df["pred_rationale_json"] = df["output_text"].apply(lambda x: parse_json_field(x, "rationale"))

    # Fallback: loose parsing for malformed or open-source model outputs
    df[["pred_label_fallback", "pred_next_fallback", "pred_rationale_fallback"]] = df["output_text"].apply(parse_loose_llm_output)

    # Combine parsed results (prefer valid JSON fields)
    df["pred_label"] = df["pred_label_json"].combine_first(df["pred_label_fallback"])
    df["pred_next_utterance"] = df["pred_next_json"].combine_first(df["pred_next_fallback"])
    df["pred_rationale"] = df["pred_rationale_json"].combine_first(df["pred_rationale_fallback"])

    # Drop intermediate columns
    df.drop(columns=[
        "pred_label_json", "pred_next_json", "pred_rationale_json",
        "pred_label_fallback", "pred_next_fallback", "pred_rationale_fallback"
    ], inplace=True, errors="ignore")

    # Save as new CSV
    output_path = str(Path(input_path).with_name(Path(input_path).stem + "_extracted.csv"))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✅ Processed: {input_path} → {output_path}")


# ===========================================================
# 6. Entry Point
# ===========================================================
if __name__ == "__main__":
    # Example input list — modify as needed
    input_files = [
        "task_SA_gpt_results.csv",
        "task_SA_gemini_results.csv",
        "task_SA_claude_results.csv",
        "SA_experiments/SA_llama8b_test_result_zeroshot.csv",
        "SA_experiments/SA_exaone2b_test_result_finetuned.csv"
    ]

    for path in input_files:
        try:
            process_csv(path)
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
