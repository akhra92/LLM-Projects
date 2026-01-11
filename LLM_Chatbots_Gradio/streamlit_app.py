"""
Python to C++ Converter - Streamlit Web Application
Converts Python code to high-performance C++ using OpenAI API.
"""

import os
import streamlit as st
from openai import OpenAI

# Page configuration
st.set_page_config(
    page_title="Python to C++ Converter",
    page_icon="🔄",
    layout="wide"
)

# System prompt for code conversion
SYSTEM_PROMPT = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""

# Example Python code
EXAMPLE_CODE = '''import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")'''


def create_user_prompt(python_code: str) -> str:
    """Create a user prompt for the code conversion."""
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
Use modern C++ (C++17 or later) with optimizations for performance.
The code will be compiled with optimization flags like -O3 or -Ofast.
Respond only with C++ code.

Python code to port:

```python
{python_code}
```
"""


def convert_to_cpp(client: OpenAI, model: str, python_code: str) -> str:
    """
    Convert Python code to C++ using OpenAI API.

    Args:
        client: OpenAI client instance
        model: Model name to use
        python_code: Python code to convert

    Returns:
        Generated C++ code
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": create_user_prompt(python_code)}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    reply = response.choices[0].message.content
    # Clean up markdown code blocks if present
    reply = reply.replace('```cpp', '').replace('```c++', '').replace('```', '').strip()
    return reply


def main():
    """Main Streamlit application."""
    st.title("🔄 Python to C++ Converter")
    st.markdown("Convert your Python code to high-performance C++ using AI")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key. It will not be stored."
        )

        # Model selection
        model = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to use for conversion"
        )

        st.divider()
        st.markdown("### About")
        st.markdown(
            "This tool uses OpenAI's GPT models to convert Python code "
            "into optimized C++ code. The generated code aims to produce "
            "identical output with better performance."
        )

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Python Code")
        python_code = st.text_area(
            "Enter your Python code:",
            value=EXAMPLE_CODE,
            height=400,
            label_visibility="collapsed"
        )

        # Load example button
        if st.button("Load Example", type="secondary"):
            st.rerun()

    with col2:
        st.subheader("C++ Code")
        cpp_placeholder = st.empty()

    # Convert button
    if st.button("Convert to C++", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
            return

        if not python_code.strip():
            st.error("Please enter some Python code to convert.")
            return

        try:
            with st.spinner("Converting Python to C++..."):
                client = OpenAI(api_key=api_key)
                cpp_code = convert_to_cpp(client, model, python_code)

            with col2:
                cpp_placeholder.code(cpp_code, language="cpp")

            # Download button
            st.download_button(
                label="Download C++ Code",
                data=cpp_code,
                file_name="main.cpp",
                mime="text/plain",
                use_container_width=True
            )

            st.success("Conversion complete!")

        except Exception as e:
            st.error(f"Error during conversion: {str(e)}")

    # Instructions
    with st.expander("How to compile and run the C++ code"):
        st.markdown("""
        After downloading the C++ code, you can compile and run it locally:

        **On macOS/Linux:**
        ```bash
        # Using clang++ (recommended for macOS)
        clang++ -std=c++17 -O3 -o main main.cpp
        ./main

        # Using g++
        g++ -std=c++17 -O3 -o main main.cpp
        ./main
        ```

        **On Windows:**
        ```bash
        # Using MSVC
        cl /O2 /EHsc main.cpp

        # Using MinGW g++
        g++ -std=c++17 -O3 -o main.exe main.cpp
        main.exe
        ```
        """)


if __name__ == "__main__":
    main()
