# Fine-tuning and Inference with LLMs

A comprehensive framework for fine-tuning and performing inference with various Large Language Models (LLMs) including BERT, Phi-3, and Qwen2 for sentiment analysis and text generation tasks.

## Overview

This project provides modular components for:
- Fine-tuning pre-trained models on custom datasets
- Running inference with both fine-tuned and pre-trained models
- Sentiment analysis using BERT
- Causal language modeling with state-of-the-art models (Phi-3, Qwen2)
- Ready-to-use pipelines for common NLP tasks

## Project Structure

```
.
├── main.py           # Main entry point with CLI interface
├── finetune.py       # Fine-tuning functions for BERT, Phi-3, and Qwen2
├── inference.py      # Inference classes for various models
├── model.py          # BERT model handler with optimizer setup
├── dataset.py        # Dataset loading and preprocessing utilities
├── trainer.py        # Custom training loop with progress tracking
├── deployer.py       # Deployment utilities for batch predictions
└── pipelines.py      # Pre-built pipelines for common tasks
```

## Installation

```bash
pip install torch transformers datasets tqdm
```

## Usage

### Command-Line Interface

#### Sentiment Analysis - Training

```bash
python main.py \
  --mode train \
  --task sentiment_analysis \
  --pretrained_model bert-base-uncased \
  --dataset_name imdb \
  --num_epochs 10 \
  --learning_rate 2e-5 \
  --batch_size 8 \
  --max_length 128 \
  --output_dir ./results \
  --device mps
```

#### Sentiment Analysis - Inference

```bash
python main.py \
  --mode inference \
  --task sentiment_analysis \
  --pretrained_model bert-base-uncased \
  --sentence "I love this movie!" \
  --max_length 128 \
  --device mps
```

#### Causal Language Modeling

```bash
python main.py \
  --task causal_LM \
  --pretrained_model microsoft/phi-3-mini-4k-instruct \
  --sentence "The weather is beautiful today" \
  --device mps
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `inference` | Mode: `train` or `inference` |
| `--task` | str | `causal_LM` | Task: `sentiment_analysis` or `causal_LM` |
| `--num_epochs` | int | 10 | Number of training epochs |
| `--learning_rate` | float | 2e-5 | Learning rate |
| `--weight_decay` | float | 1e-2 | Weight decay |
| `--batch_size` | int | 8 | Batch size per device |
| `--output_dir` | str | `./results` | Output directory |
| `--dataset_name` | str | `imdb` | Dataset name |
| `--sentence` | str | - | Input sentence for inference |
| `--pretrained_model` | str | `bert-base-uncased` | Pretrained model name |
| `--max_length` | int | 128 | Maximum token length |
| `--device` | str | `mps` | Device: `cuda`, `mps`, or `cpu` |


## Notes

- The BERT model requires fine-tuning for meaningful predictions
- Causal LMs (Phi-3, Qwen2) can be used for zero-shot inference
- Gradient checkpointing is used for memory-efficient training of large models
- FP16 training reduces memory usage and speeds up training

## Example Output

### Sentiment Analysis
```
Sentence:
I love this movie!
Sentiment Analysis:
Positive
```

### Batch Deployment
```
Review                                              Prediction
---------------------------------------------------------
I love this movie a lot! It is very interesting.    Positive
I really did not like this drama. Terrible act...   Negative
It was alright. Not great, not terrible-just a...   Negative
```

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- tqdm
