import os
import logging
from omegaconf import OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scriptmind.dataset import RawDataset
from scriptmind.trainer import ScriptmindTrainer
import setproctitle
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--omegaconf', '-oc', type=str, default='dev')
    args = parser.parse_args()
    conf = OmegaConf.load('./settings.yaml')[args.omegaconf]

    logging.info(f"STEP [1] : Preparing ScriptMind Dataset >>>>> task_SA_train.json / task_SA_dev.json")
    dataset = RawDataset(conf, logging)
    dataset.prepare_SA_dataset()

    logging.info(f"STEP [2] : Loading Backbone Model >>>>> {conf.finetune.llm_backbone}")
    setproctitle.setproctitle(f"finetuning_{conf.finetune.llm_backbone}")
    tokenizer = AutoTokenizer.from_pretrained(conf.finetune.llm_backbone)

    if conf.finetune.lora.qlora and conf.finetune.lora.enabled:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            conf.finetune.llm_backbone,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            conf.finetune.llm_backbone,
            device_map="auto"
        )

    logging.info(f"STEP [3] : Fine-tuning ScriptMind Model >>>>> {conf.finetune.llm_backbone}")
    trainer = ScriptmindTrainer(conf, logging)
    trainer(model, tokenizer)
