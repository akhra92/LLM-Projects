import torch
from dataset import DataPreprocessor, load_dataset
from model import BertModelHandler
from transformers import Trainer, TrainingArguments, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
import argparse
from time import time


def load_model(model_name):
    model_handler = BertModelHandler(model_name=model_name)
    return model_handler.get_model()


def load_train_dataset(dataset_name):
    dataset = load_dataset(dataset_name=dataset_name)
    return dataset


def tokenize_dataset(sentences, pretrained_model_name, max_length):
    tokenizer = DataPreprocessor(pretrained_model_name=pretrained_model_name, max_length=max_length)
    tokens = tokenizer.preprocess(examples=sentences)
    return tokens


def tokenize_sentence(sentence, pretrained_model_name, max_length):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    tokens = tokenizer(sentence, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    return tokens


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="inference", help="Select mode from [train, inference]")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=8, help="Per device batch size")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--dataset_name", type=str, default="imdb", help="Dataset name")
    parser.add_argument("--sentence", type=str, default="I like Artificial Intelligence", help="Sentence for inference")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--max_length", type=int, default=128, help="Max length of tokens")
    parser.add_argument("--task", type=str, default="causal_LM", help="Task name - sentiment_analysis or causal_LM")
    parser.add_argument("--device", type=str, default="mps", help="Device (cuda, mps, or cpu)")
    
    args = parser.parse_args()

    if args.task == "sentiment_analysis":
        print("Implementing Sentiment Analysis...")
        if args.mode == "train":
            print("Training started!")
            model = load_model(model_name=args.pretrained_model)
            dataset = load_train_dataset(dataset_name=args.dataset_name)
            tokens = tokenize_dataset(sentences=dataset, pretrained_model_name=args.pretrained_model, max_length=args.max_length)
            training_args = TrainingArguments(
                output_dir = args.output_dir,
                evaluation_strategy = "epoch",
                learning_rate = args.learning_rate,
                num_train_epochs = args.num_epochs,
                weight_decay = args.weight_decay,
                per_device_train_batch_size = args.batch_size,
                per_device_eval_batch_size = args.batch_size
            )
            trainer = Trainer(
                model = model,
                args = training_args,
                train_dataset = tokens["train"].shuffle(seed=42).select(range(1000)),
                eval_dataset = tokens["test"].shuffle(seed=42).select(range(500))
            )
            trainer.train()
            print("Training finished!")
        else:
            print("Inference started!")
            model = load_model(model_name=args.pretrained_model).to(args.device)
            tokens = tokenize_sentence(args.sentence, pretrained_model_name=args.pretrained_model, max_length=args.max_length)
            input_ids, attention_mask = tokens["input_ids"].to(args.device), tokens["attention_mask"].to(args.device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = outputs.logits.argmax(dim=-1)
                sentiment = "Negative" if pred == 0 else "Positive"
            print(f"Sentence:\n{args.sentence}\nSentiment Analysis:\n{sentiment}")
            print("\nInference finished!")
    else:
        print("Implementing Causal LM...")
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        inputs = f"Summarize and Classify this: {args.sentence}\nAnswer:"
        tokens = tokenizer(inputs, return_tensors="pt")
        input_ids = tokens["input_ids"].to(args.device)
        outputs = model.generate(input_ids, max_new_tokens=32)
        print(f"Prediction:\n{tokenizer.decode(outputs[0], skip_special_tokens=False)}")       