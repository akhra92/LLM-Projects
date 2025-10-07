# Tokenizer Examples

This project demonstrates different tokenization techniques and their application in sentiment analysis using the Hugging Face transformers library.

## Overview

The notebook explores:
- WordPiece tokenization (BERT)
- Byte Pair Encoding (BPE) tokenization (GPT-2)
- IMDB dataset analysis
- Sentiment classification with BERT

## Contents

### 1. Tokenization Comparison
Compares two popular tokenization methods:
- **WordPiece Tokenizer** (BERT): `bert-base-uncased`
- **BPE Tokenizer** (GPT-2): `gpt2`


## Requirements

```bash
transformers
datasets
torch
matplotlib
numpy
pandas
```

## Installation

```bash
pip install transformers datasets torch matplotlib numpy pandas
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook Tokenizer.ipynb
```

## Notes

- The BERT model in the final section is **not trained** - predictions are from randomly initialized classifier weights
- To get meaningful results, the model should be fine-tuned on the IMDB dataset
- Special tokens used: `[CLS]` (start), `[SEP]` (end)
