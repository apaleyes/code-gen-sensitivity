# to run this use from the project root
# python -m sandbox.identify_reference_code

from collections import Counter

from models import ModelCaller, get_model
from prompt_utils import ensure_python_code_prompt


model_name = "dummy"  # possible values 'openai', 'gemini', 'dummy', 'claude'
prompt = "Implement a convex hull algorithm for 4D points. It should be able to take in a list of points, validate that all of them are of four dimensions, and calculate the minimal convex hull of the given list of points. It should be able to cover edge cases, such as a set of one point, points on a single plane, and more. Define the data structure that is most sensible for output of this problem, and make sure the algorithm returns data in that structure."
n_repeats = 10

model = get_model(model_name)
model_caller = ModelCaller(model, prompt_transform=ensure_python_code_prompt)

all_codes = []
for i in range(n_repeats):
    print(f"Call #{i+1}")
    code = model_caller.get_code(prompt)
    all_codes.append(code)

most_common_code, most_common_count = Counter(all_codes).most_common(1)[0]

print(f"Most common code seen {most_common_count} out of {n_repeats}")
