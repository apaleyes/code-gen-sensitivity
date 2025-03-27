def test_ensure_python_code_prompt():
    from code_utils import ensure_python_code_prompt

    prompt = "some random prompt"

    new_prompt = ensure_python_code_prompt(prompt)

    assert prompt != new_prompt
    assert prompt in new_prompt


def test_remove_comments():
    from code_utils import remove_comments_and_docstrings

    python_code = """
def my_function():
    '''This is a docstring
    many lines
    multiline'''
    x = 10  # This is a single-line comment
    y = 20
    # Another single-line comment
    return x + y
"""

    cleaned_code = remove_comments_and_docstrings(python_code)

    assert "This is a docstring" not in cleaned_code
    assert "multiline" not in cleaned_code
    assert "single-line comment" not in cleaned_code

    assert "my_function" in cleaned_code
    assert "return x + y" in cleaned_code
