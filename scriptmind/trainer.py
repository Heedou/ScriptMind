from datasets import load_from_disk
import transformers
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import os
from omegaconf import OmegaConf

class ScriptmindTrainer(object):
    def __init__(self, conf, logging):
        self.logging = logging
        self.conf = conf

        # Dataset path
        base_path = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.path_train = os.path.join(base_path, f"task_SA_train_{conf.dataprep.subsample}.hf")
        self.path_dev   = os.path.join(base_path, f"task_SA_dev_{conf.dataprep.subsample}.hf")

        # Checkpoint
        wandb_name = f"{conf.wandb.project_name}_{conf.wandb.group_name}_{conf.wandb.session_name}/"
        self.checkpoint_path = os.path.join(conf.path.checkpoint, wandb_name)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        OmegaConf.save(config=conf, f=os.path.join(self.checkpoint_path, 'config.yaml'))

        # LORA config
        self.llm_backbone = conf.finetune.llm_backbone
        lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.lora_config = LoraConfig(
            r=conf.finetune.lora.r,
            lora_alpha=conf.finetune.lora.alpha,
            target_modules=lora_modules,
            lora_dropout=conf.finetune.lora.dropout,
            bias=conf.finetune.lora.bias,
            task_type=conf.finetune.lora.task_type
        )

        # Training config
        self.train_config = transformers.TrainingArguments(
            per_device_train_batch_size=conf.finetune.per_device_train_batch_size,
            gradient_accumulation_steps=conf.finetune.gradient_accumulation_steps,
            num_train_epochs=conf.finetune.num_train_epochs,
            learning_rate=conf.finetune.learning_rate,
            fp16=conf.finetune.fp16,
            logging_steps=conf.finetune.logging_steps,
            output_dir=self.checkpoint_path,
            optim=conf.finetune.optim,
            report_to=conf.finetune.report_to,
            save_strategy="epoch",
            evaluation_strategy="epoch"
        )

    @torch.no_grad()
    def print_trainable_parameters(self, model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        self.logging.info(f"Trainable: {trainable} / Total: {total} ({100 * trainable / total:.2f}%)")

    def __call__(self, model, tokenizer):
        self.logging.info("Preparing ScriptMind Trainer...")
        model.gradient_checkpointing_enable()
        if self.conf.finetune.lora.enabled:
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, self.lora_config)
        self.print_trainable_parameters(model)
        tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        train_dataset = load_from_disk(self.path_train)
        eval_dataset  = load_from_disk(self.path_dev)
        train_dataset = train_dataset.map(lambda s: tokenizer(s["text"]), batched=True)
        eval_dataset  = eval_dataset.map(lambda s: tokenizer(s["text"]), batched=True)

        trainer = transformers.Trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=self.train_config,
            data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        model.config.use_cache = False
        trainer.train()
        tokenizer.save_pretrained(self.checkpoint_path)
        trainer.save_model()
        return model, trainer
