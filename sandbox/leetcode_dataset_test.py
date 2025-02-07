# to run, go to from base and do
#     python -m sandbox.leetcode_dataset_test
# uses leetcode dataset from
# https://huggingface.co/datasets/NyanDoggo/leetcode

import json

from models import ModelCaller, get_model
from utils import ensure_python_code_prompt

if __name__ == "__main__":
    with open("sandbox/leetcode-dataset.json", "r") as f:
        leetcode_data = json.load(f)

    model = get_model("gemini")
    model_caller = ModelCaller(model, prompt_transform=ensure_python_code_prompt)

    for entry in leetcode_data[10:13]:
        code = model_caller.get_code(entry["question"])
        print(code)
