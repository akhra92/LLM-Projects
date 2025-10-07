#!/bin/bash

# Simple script for training and inference with LLMs

# Default settings
DEVICE="mps"
SENTENCE="I love this movie!"

# Help function
show_help() {
    echo "Usage: ./run.sh [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train-bert           Train BERT model"
    echo "  inference-bert       Run BERT inference"
    echo "  train-phi            Fine-tune Phi-3"
    echo "  train-qwen           Fine-tune Qwen2"
    echo "  inference-causal     Run causal LM inference"
    echo "  deploy               Run batch deployment"
    echo ""
    echo "Options:"
    echo "  --sentence TEXT      Input sentence for inference"
    echo "  --device DEVICE      Device: cuda, mps, or cpu (default: mps)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh train-bert"
    echo "  ./run.sh inference-bert --sentence 'This movie is great!'"
    echo "  ./run.sh inference-causal --sentence 'Explain AI' --device cpu"
}

# Parse arguments
COMMAND=$1
shift || { show_help; exit 0; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --sentence) SENTENCE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Execute commands
case $COMMAND in
    train-bert)
        echo "Training BERT..."
        python main.py --mode train --task sentiment_analysis --num_epochs 10 --batch_size 8 --device "$DEVICE"
        ;;

    inference-bert)
        echo "Running BERT inference on: $SENTENCE"
        python main.py --mode inference --task sentiment_analysis --sentence "$SENTENCE" --device "$DEVICE"
        ;;

    train-phi)
        echo "Fine-tuning Phi-3..."
        python -c "from finetune import finetune_phi; finetune_phi()"
        ;;

    train-qwen)
        echo "Fine-tuning Qwen2..."
        python -c "from finetune import finetune_qwen; finetune_qwen()"
        ;;

    inference-causal)
        echo "Running causal LM inference on: $SENTENCE"
        python main.py --task causal_LM --sentence "$SENTENCE" --device "$DEVICE"
        ;;

    deploy)
        echo "Running batch deployment..."
        python deployer.py
        ;;

    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "Done!"
