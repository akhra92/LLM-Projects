"""
Python to C++ Converter - Converts Python code to high-performance C++ using OpenAI API
This tool uses GPT to port Python code to optimized C++ and compiles/runs it.
"""

import os
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from system_info_ret import retrieve_system_info

# Load environment variables
load_dotenv(override=True)

# OpenAI configuration
OPENAI_MODEL = "gpt-4o"

# Compile commands for C++
COMPILE_COMMAND = [
    "clang++", "-std=c++17", "-Ofast", "-mcpu=native",
    "-flto=thin", "-fvisibility=hidden", "-DNDEBUG",
    "main.cpp", "-o", "main"
]
RUN_COMMAND = ["./main"]

# System prompt for code conversion
SYSTEM_PROMPT = """
Your task is to convert Python code into high performance C++ code.
Respond only with C++ code. Do not provide any explanation other than occasional comments.
The C++ response needs to produce an identical output in the fastest possible time.
"""


def get_system_info() -> dict:
    """Retrieve system information for optimized compilation."""
    return retrieve_system_info()


def create_user_prompt(python_code: str, system_info: dict) -> str:
    """
    Create a user prompt for the code conversion.

    Args:
        python_code: The Python code to convert
        system_info: System information dictionary

    Returns:
        Formatted prompt string
    """
    return f"""
Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
The system information is:
{system_info}
Your response will be written to a file called main.cpp and then compiled and executed; the compilation command is:
{COMPILE_COMMAND}
Respond only with C++ code.
Python code to port:

```python
{python_code}
```
"""


def create_messages(python_code: str, system_info: dict) -> list:
    """
    Create the message list for the OpenAI API call.

    Args:
        python_code: The Python code to convert
        system_info: System information dictionary

    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": create_user_prompt(python_code, system_info)}
    ]


def write_cpp_output(cpp_code: str, filename: str = "main.cpp") -> None:
    """
    Write C++ code to a file.

    Args:
        cpp_code: The C++ code to write
        filename: Output filename (default: main.cpp)
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(cpp_code)


def port_python_to_cpp(client: OpenAI, model: str, python_code: str) -> str:
    """
    Port Python code to C++ using OpenAI API.

    Args:
        client: OpenAI client instance
        model: Model name to use
        python_code: Python code to convert

    Returns:
        Generated C++ code
    """
    system_info = get_system_info()
    response = client.chat.completions.create(
        model=model,
        messages=create_messages(python_code, system_info)
    )
    reply = response.choices[0].message.content
    # Clean up markdown code blocks if present
    reply = reply.replace('```cpp', '').replace('```', '').strip()
    write_cpp_output(reply)
    return reply


def run_python(code: str) -> None:
    """
    Execute Python code directly.

    Args:
        code: Python code string to execute
    """
    globals_dict = {"__builtins__": __builtins__}
    exec(code, globals_dict)


def compile_and_run(num_runs: int = 3) -> list:
    """
    Compile the C++ code and run it multiple times.

    Args:
        num_runs: Number of times to run the compiled program

    Returns:
        List of output strings from each run
    """
    # Compile
    compile_result = subprocess.run(
        COMPILE_COMMAND,
        check=True,
        text=True,
        capture_output=True
    )

    # Run multiple times
    outputs = []
    for _ in range(num_runs):
        run_result = subprocess.run(
            RUN_COMMAND,
            check=True,
            text=True,
            capture_output=True
        )
        outputs.append(run_result.stdout)
        print(run_result.stdout)

    return outputs


# Example Python code for testing
EXAMPLE_PI_CODE = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""


def main():
    """Main function to demonstrate the Python to C++ conversion."""
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    print("OpenAI API Key exists.")

    # Initialize client
    client = OpenAI()

    # Get system info
    system_info = get_system_info()
    print(f"\nSystem Info: {system_info}\n")

    # Run Python version
    print("Running Python version:")
    print("-" * 40)
    run_python(EXAMPLE_PI_CODE)
    print("-" * 40)

    # Port to C++
    print("\nConverting to C++...")
    cpp_code = port_python_to_cpp(client, OPENAI_MODEL, EXAMPLE_PI_CODE)
    print("C++ code written to main.cpp")

    # Compile and run C++ version
    print("\nRunning C++ version (3 runs):")
    print("-" * 40)
    compile_and_run(num_runs=3)
    print("-" * 40)


if __name__ == "__main__":
    main()
