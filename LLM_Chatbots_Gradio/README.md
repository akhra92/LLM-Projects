# LLM Chatbot Gradio And Python to C++ Converter

LLM-powered tools (Interviewer & C++ Converter) using Gradio and Streamlit interfaces.

## Contents

- **LLM_as_Interviewer.py** - AI engineer interview simulator using Ollama (llama3.2) with Gradio
- **streamlit_app.py** - Web UI for Python to C++ conversion using OpenAI API
- **python_to_cpp.py** - CLI tool to convert Python to optimized C++ with auto-compilation
- **system_info_ret.py** - System info utility for optimized C++ compilation flags

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Interview simulator
python LLM_as_Interviewer.py

# Streamlit converter
streamlit run streamlit_app.py

# CLI converter (requires OPENAI_API_KEY env var)
python python_to_cpp.py
```
