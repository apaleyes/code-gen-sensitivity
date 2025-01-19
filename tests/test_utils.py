def test_ensure_python_code_prompt():
    from utils import ensure_python_code_prompt

    prompt = "some random prompt"

    new_prompt = ensure_python_code_prompt(prompt)

    assert prompt != new_prompt
    assert prompt in new_prompt
