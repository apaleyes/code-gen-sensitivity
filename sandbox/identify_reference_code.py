# to run this use from the project root
# python -m sandbox.identify_reference_code

from collections import Counter

from models import ModelCaller, get_model
from code_utils import ensure_python_code_prompt, remove_comments_and_docstrings


model_name = "openai"  # possible values 'openai', 'gemini', 'dummy', 'claude'
prompt_convex_hull = "Implement a convex hull algorithm for 4D points. It should be able to take in a list of points, validate that all of them are of four dimensions, and calculate the minimal convex hull of the given list of points. It should be able to cover edge cases, such as a set of one point, points on a single plane, and more. Define the data structure that is most sensible for output of this problem, and make sure the algorithm returns data in that structure."
prompt_streaming_service = "Design and implement back end of a video streaming web site. It should provide all operations for users to watch videos online. It should also be able to deal with videos of small and large size, support streaming in different resolutions, give users the ability to pause, restart, skip forwards and backwards. Take into account risks of variable connection quality, potentially very large upload and download sizes, and concurrency."
n_repeats = 20

model = get_model(model_name)
model_caller = ModelCaller(model, prompt_transform=ensure_python_code_prompt)

all_codes = []
for i in range(n_repeats):
    print(f"Call #{i+1}")
    code = model_caller.get_code(prompt_streaming_service)
    code = remove_comments_and_docstrings(code)
    all_codes.append(code)

most_common_code, most_common_count = Counter(all_codes).most_common(1)[0]

print(f"Most common code seen {most_common_count} out of {n_repeats}")
