# helper.py

import random
import time
import concurrent.futures
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from TSED import TSED
import code_execute


def get_augmenter(method, aug_rate, prompt_len):
    if method == "keyboard":
        return nac.KeyboardAug(
            aug_char_p=aug_rate,
            aug_word_p=aug_rate,
            aug_char_min=0,
            aug_word_max=prompt_len
        )
    elif method == "synonym":
        return naw.SynonymAug(
            aug_p=aug_rate,
            aug_max=prompt_len
        )
    else:
        raise ValueError(f"Unknown augmentation method: {method}")


def generate_augmented_prompts(original_prompt, augmenter, n):
    return [augmenter.augment(original_prompt, n=1)[0] for _ in range(n)]


def call_with_retry(model_caller, prompt, retries=3, timeout=30):
    time.sleep(random.randint(0, 10))
    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_ = executor.submit(model_caller.get_code, prompt)
                return future_.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Timeout. Retry {attempt+1}/{retries}")
            if attempt == retries - 1:
                raise
            time.sleep(1)


def calculate_metrics(original_code, new_code, gt_solution=None, prompt=None):
    try:
        sol_sim, acc = None, None
        code_sim = TSED.Calaulte("python", original_code, new_code, 1.0, 0.8, 1.0)
        if gt_solution is not None:
            sol_sim = TSED.Calaulte("python", gt_solution, new_code, 1.0, 0.8, 1.0)
            acc = code_execute.evaluate_solution(new_code, prompt)
        return code_sim, sol_sim, acc
    except Exception as e:
        print("Metric calculation failed:", e)
        return None, None, None
