def ensure_python_code_prompt(prompt: str):
    prefix = "Write Python code."
    postfix = "Output only Python code and nothing else. CRITICAL:Do not include any markdown _or_ code block indicators."
    prompt = prefix + prompt + postfix

    return prompt
