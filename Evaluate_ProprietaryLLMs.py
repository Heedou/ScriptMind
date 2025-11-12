"""
===========================================================
Voice Phishing Detection Evaluation Script
===========================================================

This script evaluates Korean voice phishing conversations using
Large Language Models (LLMs) from multiple providers:

  - OpenAI GPT models
  - Google Gemini
  - Anthropic Claude

Each model produces structured JSON outputs for conversation
classification ("vishing" / "non_vishing") and reasoning explanation.

-----------------------------------------------------------
Usage:
    python evaluate_vishing.py --model [gpt|gemini|claude]

Before running:
    export OPENAI_API_KEY="your_openai_key"
    export GEMINI_API_KEY="your_gemini_key"
    export ANTHROPIC_API_KEY="your_claude_key"

Input File:
    task_SA_test.json

Output:
    task_SA_[model]_results.csv
-----------------------------------------------------------
"""

import os
import json
import csv
import time
import random
import argparse
from tqdm import tqdm

# Optional imports handled safely
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from anthropic import Anthropic, APIError, APIStatusError
except ImportError:
    Anthropic = None
    APIError = None
    APIStatusError = None


# ===========================================================
# Shared Prompt Template
# ===========================================================

def build_prompt(conversation: str) -> str:
    """
    Build a unified evaluation prompt for voice phishing detection.
    """
    return f"""
You are an expert in detecting Korean voice phishing (vishing) conversations.
You must return a single valid JSON object only. No additional text or commentary.

Below is the given conversation:
---
{conversation}
---

Follow these rules:
- If the dialogue indicates voice phishing, return JSON with label:"vishing" and fill in both next_utterance and rationale.
- If it is NOT voice phishing, return only {{"label":"non_vishing"}}.

Example output (for phishing):
{{
  "label": "vishing",
  "next_utterance": "Predicted next sentence(s) from the scammer (1–2 lines).",
  "rationale": "Current scammer’s intent: ... . Expected next intent: ... . Reasoning: ..."
}}

Now analyze the given conversation and produce the result strictly in JSON format.  
⚠️ You MUST output VALID JSON ONLY. No extra text, explanations, or formatting.
"""


# ===========================================================
# Model-Specific Evaluation Functions
# ===========================================================

def run_gpt(data):
    """Run evaluation using OpenAI GPT models."""
    if OpenAI is None:
        raise ImportError("OpenAI package not found. Please install with `pip install openai`.")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model_name = "chatgpt-4o-latest"
    output_file = "task_SA_gpt_results.csv"

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["conversation", "vishing", "source", "output", "output_text"])
        writer.writeheader()

        for conv_id, conv_data in tqdm(data.items()):
            conversation = conv_data["conversation"]
            vishing = conv_data.get("vishing", "")
            source = conv_data.get("source", "")
            output = json.dumps(conv_data.get("output", {}), ensure_ascii=False)

            system_prompt = (
                "You are a Korean voice phishing detection expert. "
                "You must return a single valid JSON object only. Do not include any other text."
            )
            user_prompt = build_prompt(conversation)

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )

            output_text = response.choices[0].message.content.strip()

            writer.writerow({
                "conversation": conversation,
                "vishing": vishing,
                "source": source,
                "output": output,
                "output_text": output_text
            })

    print(f"✅ GPT evaluation completed. Results saved to {output_file}")


def run_gemini(data):
    """Run evaluation using Google Gemini."""
    if genai is None:
        raise ImportError("google-generativeai package not found. Install with `pip install google-generativeai`.")

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    output_file = "task_SA_gemini_results.csv"

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["conversation", "vishing", "source", "output", "output_text"])
        writer.writeheader()

        for conv_id, conv_data in tqdm(data.items()):
            conversation = conv_data["conversation"]
            vishing = conv_data.get("vishing", "")
            source = conv_data.get("source", "")
            output = json.dumps(conv_data.get("output", {}), ensure_ascii=False)
            prompt = build_prompt(conversation)

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=512,
                ),
            )

            output_text = response.text.strip() if response.text else ""

            writer.writerow({
                "conversation": conversation,
                "vishing": vishing,
                "source": source,
                "output": output,
                "output_text": output_text
            })

    print(f"✅ Gemini evaluation completed. Results saved to {output_file}")


def run_claude(data):
    """Run evaluation using Anthropic Claude."""
    if Anthropic is None:
        raise ImportError("anthropic package not found. Install with `pip install anthropic`.")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model_name = "claude-3-5-haiku-20241022"
    output_file = "task_SA_claude_results.csv"

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["conversation", "vishing", "source", "output", "output_text"])
        writer.writeheader()

        for conv_id, conv_data in tqdm(data.items()):
            conversation = conv_data["conversation"]
            vishing = conv_data.get("vishing", "")
            source = conv_data.get("source", "")
            output = json.dumps(conv_data.get("output", {}), ensure_ascii=False)
            prompt = build_prompt(conversation)

            max_retries = 3
            output_text = ""

            for attempt in range(max_retries):
                try:
                    response = client.messages.create(
                        model=model_name,
                        max_tokens=512,
                        temperature=0.2,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    output_text = response.content[0].text.strip()
                    break
                except APIStatusError as e:
                    if e.status_code == 529:
                        wait_time = random.uniform(3, 8)
                        print(f"⚠️ Claude overloaded: retrying in {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ APIStatusError ({e.status_code}): {e.message}")
                        output_text = f"APIStatusError: {e.message}"
                        break
                except APIError as e:
                    print(f"❌ APIError: {e}")
                    output_text = f"APIError: {e}"
                    break

            writer.writerow({
                "conversation": conversation,
                "vishing": vishing,
                "source": source,
                "output": output,
                "output_text": output_text
            })

            # Prevent overloading
            time.sleep(random.uniform(0.5, 1.5))

    print(f"✅ Claude evaluation completed. Results saved to {output_file}")


# ===========================================================
# Main Entry Point
# ===========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Korean voice phishing conversations using multiple LLMs.")
    parser.add_argument("--model", choices=["gpt", "gemini", "claude"], required=True,
                        help="Model type to use: gpt | gemini | claude")
    args = parser.parse_args()

    with open("task_SA_test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.model == "gpt":
        run_gpt(data)
    elif args.model == "gemini":
        run_gemini(data)
    elif args.model == "claude":
        run_claude(data)
