# Fine-tuning & RAG with LoRA

A comprehensive implementation of advanced deep learning techniques including LoRA/QLoRA fine-tuning, Retrieval-Augmented Generation (RAG), and Vision Transformers (ViT).

## Overview

This project demonstrates practical implementations of:
- **LoRA/QLoRA**: Parameter-efficient fine-tuning with low-rank adaptation
- **Qwen Model Fine-tuning**: Fine-tuning Qwen language models using LoRA on AG News dataset
- **RAG**: Retrieval-augmented generation for context-aware LLM responses
- **Vision Transformer (ViT)**: Complete implementation of transformer architecture for image classification
- **FastAPI Deployment**: Production-ready API for serving fine-tuned models

## Features

### LoRA & QLoRA (`lora.py`)
- Custom implementation of Low-Rank Adaptation (LoRA) technique
- Quantized LoRA (QLoRA) with 8-bit quantization for memory efficiency
- Demonstrates significant memory reduction while maintaining model performance
- Comparative analysis between LoRA and QLoRA approaches

### Qwen Fine-tuning (`qwen_lora.py`)
- Fine-tunes Qwen2-1.5B model using PEFT library
- Trains on AG News dataset for text classification
- Configurable LoRA parameters (rank, alpha, dropout)
- Supports quantization with BitsAndBytes integration

### RAG System (`rag.py`)
- Document retrieval based on cosine similarity
- Query encoding and context-aware generation
- Extensible knowledge base system
- Combined retrieval and generation pipeline

### Vision Transformer (`vit.py`)
- Complete ViT architecture from scratch
- Patch embedding with convolutional projection
- Multi-head self-attention mechanism
- Position embeddings and classification head

### Model Deployment (`app.py`)
- FastAPI-based REST API for model inference
- Loads fine-tuned PEFT models
- Configurable generation parameters
- Ready for production deployment

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Finetuning_RAG_LORA

# Install dependencies
pip install torch transformers peft datasets accelerate bitsandbytes fastapi uvicorn pydantic
```

## Usage

### Training Qwen with LoRA

```bash
python qwen_lora.py
```

This will:
- Load Qwen2-1.5B base model
- Apply LoRA adapters (rank=16, alpha=32)
- Fine-tune on 1000 samples from AG News dataset
- Save the fine-tuned model to `./qwen2_peft_result`

### Testing LoRA/QLoRA

```bash
python lora.py
```

Compares memory consumption and demonstrates parameter efficiency between LoRA and QLoRA approaches.

### Running RAG Demo

```bash
python rag.py
```

Demonstrates document retrieval and context-aware generation with a sample knowledge base.

### Vision Transformer

```bash
python vit.py
```

Runs inference with a ViT model on 384x384 images with 10 output classes.

### Deploying the API

```bash
python app.py
```

Starts a FastAPI server on `http://0.0.0.0:8001` with endpoints:
- `GET /`: Health check
- `POST /generate`: Generate text from prompts

**Example request:**
```bash
curl -X POST "http://localhost:8001/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "max_new_token": 50}'
```

## Project Structure

```
.
├── app.py              # FastAPI deployment server
├── lora.py             # LoRA & QLoRA implementation
├── qwen_lora.py        # Qwen model fine-tuning script
├── rag.py              # RAG implementation
├── vit.py              # Vision Transformer implementation
└── qwen2_peft_result/  # Fine-tuned model checkpoints
```

## Model Configuration

### LoRA Parameters
- **Rank (r)**: 16
- **Alpha**: 32
- **Target modules**: `q_proj`, `v_proj`
- **Dropout**: 0.1
- **Task type**: Causal Language Modeling

### Training Arguments
- **Epochs**: 10
- **Batch size**: 2
- **Learning rate**: 1e-4
- **Strategy**: Save per epoch

## Technical Details

### Memory Efficiency
- **LoRA**: Reduces trainable parameters by freezing base model weights and training low-rank matrices
- **QLoRA**: Further reduces memory with 8-bit quantization of base model weights
- **Typical reduction**: 75%+ memory savings compared to full fine-tuning

### RAG Pipeline
1. Encode user query into embedding space
2. Retrieve top-k relevant documents using cosine similarity
3. Combine query with retrieved context
4. Generate response using language model

### ViT Architecture
- **Patch size**: 16x16
- **Embedding dimension**: 768
- **Attention heads**: 12
- **Transformer blocks**: 10
- **MLP ratio**: 4.0

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT
- Datasets
- FastAPI
- Uvicorn

## Notes

- The Qwen models require `trust_remote_code=True` for loading
- Fine-tuning requires sufficient GPU memory (8GB+ recommended)
- QLoRA enables training on consumer GPUs by reducing memory footprint
- The API server in `app.py` has a typo: should be `uvicorn.run(api, ...)` instead of `app`

## License

MIT
