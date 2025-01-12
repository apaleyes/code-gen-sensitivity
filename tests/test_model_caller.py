import pytest

from models import BaseModel, ModelCaller, get_model

def test_correct_model():
    model = get_model("dummy")
    caller = ModelCaller(model)

    result = caller.get_code("test prompt")

    assert isinstance(result, str)
    assert len(result) > 0


def test_model_returns_invalid_python_code():
    code = "a = 1 + 2"
    class WrappedCodeModel(BaseModel):
        def __init__(self):
            super().__init__("WrappedCodeModel")
        
        def __call__(self, prompt):
            return "\n".join(["```python", code, "```"])

    model = WrappedCodeModel()
    caller = ModelCaller(model)
    result = caller.get_code("test prompt")

    assert code == result


def test_model_after_retries():
    code = "a = 1 + 2"
    class ModelWithRetries(BaseModel):
        def __init__(self):
            super().__init__("ModelWithRetries")
            self.count = 0
        
        def __call__(self, prompt):
            if self.count < 3:
                self.count += 1
                raise ValueError("testing retries")
            return "a = 1 + 2"

    model = ModelWithRetries()
    caller = ModelCaller(model)
    result = caller.get_code("test prompt")

    assert code == result


def test_model_returns_invalid_python_code():
    class InvalidCodeModel(BaseModel):
        def __init__(self):
            super().__init__("InvalidCodeModel")
        
        def __call__(self, prompt):
            return "df blah"

    model = InvalidCodeModel()
    caller = ModelCaller(model)

    with pytest.raises(RuntimeError):
        caller.get_code("test prompt")
