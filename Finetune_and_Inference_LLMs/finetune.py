from dataset import LoadDataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def finetune_bert():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    dataset = LoadDataset()
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length', batched=True))

    args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01
    )

    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset['train'].shuffle(seed=42).select(range(2000)),
        eval_dataset=tokenized_dataset['test'].shuffle(seed=42).select(range(500))
        
    )

    trainer.train()


def finetune_phi():
    model_id = 'microsoft/phi-3-mini-4k-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.gradient_checkpointing_enable()

    dataset = load_dataset('yelp_review_full', split='train[:500]')

    def tokenize_function(example):
        prompt = f"Review: {example['text']}\nSentiment:"
        tokenized = tokenizer(prompt, truncation=True, padding='max_length', max_length=8)
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dirs='./results',
        per_device_train_batch_size=1,
        num_train_epochs=1,
        evaluation_strategy='no',
        save_strategy='no',
        learning_rate=1e-5,
        fp16=True

    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=tokenized_dataset,
        tokenizer = tokenizer,
        data_collector=None             # No dynamic padding needed with fixed max_length
    )

    trainer.train()



def finetune_qwen():
    model_id = 'Qwen/Qwen2-1.5B'
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    model.gradient_checkpointing_enable() # For memory optimization

    dataset = load_dataset('yelp_review_full', split='train[:500]')

    def tokenize_function(example):
        prompt = f"Review: {example['text']}\nSentiment:"
        tokenized = tokenizer(prompt, truncation=True, padding='max_length', max_length=128)
        # for causal LM full fine-tuning, labels are same as input_ids
        tokenized["labels"] = tokenized["input_ids"]
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_function, batched=False)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dirs='./full_finetuned_qwen',
        per_device_train_batch_size=1,
        num_train_epochs=3,
        evaluation_strategy='no',
        save_strategy='no',
        learning_rate=5e-5,
        fp16=True,
        logging_steps=50

    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=tokenized_dataset,
        tokenizer = tokenizer,
        data_collator=None
    )

    trainer.train()


