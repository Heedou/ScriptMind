# ScriptMind

---
## 📂 Input Data Format

The input file should be named **`task_SA_test.json`**,  
and must follow the structure shown below.

Each entry represents one conversation sample with its metadata.

```json
{
  "0": {
    "conversation": "Victim: Hello? / Scammer: This is from the National Tax Service...",
    "vishing": "vishing",
    "source": "dataset_A",
    "output": {}
  },
  "1": {
    "conversation": "Caller: Your bank account needs verification...",
    "vishing": "non_vishing",
    "source": "dataset_B",
    "output": {}
  }
}
```

---
The dataset used in this research is available via the link below.

- **Dataset Link:** [🔗 Download ](https://drive.google.com/file/d/1WGJId_BOp9b0q3-VYn5etlTP9gXu1lox/view?usp=drive_link)

> ⚠️ **Note:** This dataset is provided for research and non-commercial use only.  
> Please cite the original authors and this project when using the dataset.

---
## 🚀 Running the Evaluation on Proprietary Models

After setting up your environment and API keys,  
you can run the evaluation script by specifying which model to use.

The command-line argument `--model` accepts one of the following options:
- `gpt` → OpenAI GPT (chatgpt-4o-latest)
- `gemini` → Google Gemini (gemini-2.0-flash-001)
- `claude` → Anthropic Claude (claude-3-5-haiku-20241022)

---

### 🧠 OpenAI GPT

Run evaluation using **ChatGPT (4o-latest)**:

```bash
python evaluate_vishing.py --model gpt
```
