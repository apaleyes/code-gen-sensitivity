import ast
import re
import unicodedata


def code_warning_detector(text_code):
    normalized_code = unicodedata.normalize("NFKC", text_code)
    # comments are safe
    normalized_code = "\n".join(line for line in normalized_code.split("\n") if not line.strip().startswith("#"))

    blocked_list = {
        "exec", "eval", "compile", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "os", "subprocess", "shutil",
        "sys", "ctypes", "cffi", "threading", "multiprocessing", "fcntl",
        "open", "write", "read", "delete", "unlink", "rename",
        "rmdir", "mkdir", "rmtree", "pathlib", "socket", "requests",
        "urllib", "http", "ftplib", "paramiko", "telnetlib", "asyncio",
        "twisted", "base64", "marshal", "pickle", "cryptography",
        "pycryptodome", "hashlib", "zlib", "memoryview", "buffer", "super",
        "execfile", "id", "object", "gc", "inspect", "breakpoint", "help"
    }
    # past blocked list: "remove", "del"

    # Use regex to ensure only full words (not substrings) are matched
    found_unsafe = set()
    for word in blocked_list:
        pattern = rf"\b{re.escape(word)}\b|\b{re.escape(word)}[.\//]"
        if re.search(pattern, normalized_code):
            found_unsafe.add(word)

    # If any unsafe keywords are found, warn the user
    if found_unsafe:
        raise RuntimeError("🚨 Unsafe code detected! Halting execution.")
        print(text_code, flush=True)  # Show the detected code
        print(f"⚠️ Potentially unsafe code detected: {', '.join(found_unsafe)}", flush=True)
        response = input("Do you confirm this code is safe? (Type 'safe' to continue): ").strip().lower()
        if response[:4] != "safe":
            raise RuntimeError("🚨 Unsafe code detected! Halting execution.")

    return text_code


def generate_test_cases(task_desc):
    test_cases = []
    lines = task_desc.strip().split("\n")

    for i in range(len(lines) - 2):  # Ensure there are at least 2 more lines ahead
        if lines[i].strip().startswith("Example"):
            if lines[i + 1].strip().startswith("Input") and lines[i + 2].strip().startswith("Output"):
                input_line = (lines[i + 1].strip()[7:])  # Print Input line
                input_line = re.sub(r"\b\w+\s*=\s*(?!\[|\()", "", input_line)
                output_line = lines[i + 2].strip()[8:]  # Print Output line
                input_eval = ast.literal_eval(input_line)
                output_eval = ast.literal_eval(output_line)
                test_cases.append((input_eval, output_eval))
    return test_cases


def has_input_calls(code):
    """Check if the function contains any calls to input()."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "input":
                return True
    except Exception as e:
        pass
        # print(f"AST Parsing Error: {e}", flush=True)
    return False

def evaluate_function(func_str_, test_cases_):
    if has_input_calls(func_str_):
        # print("Function contains input() calls. Skipping execution.", flush=True)
        return 0  # Consider input-blocking functions as failed

    global_scope = {}  # Shared scope to execute the entire script

    try:
        exec(func_str_, global_scope)  # Execute the entire function string

        func_names = re.findall(r"def (\w+)\(", func_str_)
        if not func_names:
            # print("No functions found!", flush=True)
            return 0

        best_score = 0
        best_func = None

        for func_name in func_names:
            if func_name in global_scope:
                func = global_scope[func_name]  # Get function from global scope
                correct = 0

                for inputs, expected in test_cases_:
                    if not isinstance(inputs, tuple):
                        inputs = (inputs,)
                    try:
                        result = func(*inputs)
                        # print(f"Function: {func_name}, Result: {result}, Expected: {expected}", flush=True)
                        if result == expected:
                            correct += 1
                    except Exception as e:
                        pass
                        # print(f"Error in function '{func_name}' with input {inputs}: {e}", flush=True)

                score = correct / len(test_cases_)
                if score > best_score:
                    best_score = score
                    best_func = func_name

        # print(f"Best function: {best_func} with score: {best_score}", flush=True)
        return best_score

    except Exception as e:
        print(f"Execution Error: {e}", flush=True)
        return 0



def remove_type_annotations(func_str):
    lines = func_str.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            # Remove return type annotations (e.g., `-> List[int]:` becomes `:`)
            line = re.sub(r"\s*->\s*[^:]+:", ":", line)
            # Remove parameter type annotations (e.g., `nums: List[int]` becomes `nums`)
            line = re.sub(r"(\w+)\s*:\s*[^,)=]+", r"\1", line)
            lines[i] = line  # Update the line in the list

    return "\n".join(lines)


def evaluate_solution(func_def_, task_desc_):
    try:
        func_def_ = remove_type_annotations(func_def_)
        func_def = code_warning_detector(func_def_)
        test_cases = generate_test_cases(task_desc_)
        return evaluate_function(func_def, test_cases)
    except Exception as e:
        # print(func_def_, e, flush=True)
        print('FAILED EVAL: score 0.0', flush=True)
        return 0.0
