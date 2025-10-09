import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset


class QwenLora:
    def __init__(self, model_id, output_dir, base_model, tokenizer, bnb_config, dataset_split="train[:1000]"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.bnb_config = bnb_config
        self.dataset_split = dataset_split
        self.lora_config = LoraConfig(
            r = 16,
            lora_alpha = 32,
            target_modules = ["q_proj", 'v_proj'],
            lora_dropout = 0.1,
            bias = "none",
            task_type = TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(base_model, self.lora_config)
        dataset = load_dataset("ag_news", split = self.dataset_split)
        tokenized_data = dataset.map(self.preprocess, batched=False)
        tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        self.dataset = tokenized_data
        
    
    def preprocess(self, example):
        tokens = self.tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    

    def train(self, epochs=10, batch_size=2, learning_rate=1e-4):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            save_strategy="epoch",
            eval_strategy="no",
        )
        trainer = Trainer(
            args = training_args,
            model = self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            data_collator=None
        )
        trainer.train()
        trainer.save_model(self.output_dir)


model_id = "Qwen/Qwen2-1.5B"
output_dir = "./qwen2_peft_result"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
trainer = QwenLora(model_id=model_id, output_dir=output_dir, base_model=model, tokenizer=tokenizer, bnb_config=None)
trainer.train()