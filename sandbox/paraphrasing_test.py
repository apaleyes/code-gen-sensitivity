import json
import os
import warnings
from datetime import datetime
from itertools import product
from typing import Dict, List

import nltk
import pandas as pd
import torch
from paraphraser_evaluation import ParaphraseEvaluator
from llm_paraphraser import LLMParaphraser
from parrot import Parrot
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration, BartTokenizer,
                          PegasusForConditionalGeneration, PegasusTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer,
                          BartForConditionalGeneration, BartTokenizer)
from dotenv import load_dotenv
load_dotenv()


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download("punkt")

# Test phrases
test_phrases = [
    "Write Python code for addition, subtraction, division, multiplication, and other similar operations, all a part of one class",
    "Given the business logic code below, implement Flask backend. Do not include example of usage or the business logic, do not repeat any code from this prompt. Only write the Flask API code.",
    "Write a Calculator class. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry."
]

# Approaches and models for parapharsing
approaches = ["transformers", "llms"]
models = [
    "Vamsi/T5_Paraphrase_Paws",    # Transformers models
    "facebook/bart-base",
    "t5-base",
    "tuner007/pegasus_paraphrase",
    #"claude",                       # LLM models
    #"openai",
    "gemini",
    "llama",
    "deepseek"
]


# Define parameter grid for experiments with transformers
param_grid_transformers = {
    "temperature": [0.0, 0.5, 1.0],
    "repetition_penalty": [1.5],
    "top_p": [0.95]
}


# Define parameter grid for experiments with LLMs
param_grid_llms = {
    "temperature": [0.0, 0.5, 1.0, 1.5, 2.0],
}

# Creates tokenizers and models for paraphrasing
def paraphrasing_factory(model_name):
    """
    This function initializes and returns a tokenizer and a model based on the model name provided.
    It supports four models: Vamsi/T5_Paraphrase_Paws, facebook/bart-base, t5-base, and tuner007/pegasus_paraphrase.
    If an unsupported model name is provided, it raises an exception.
    """
    if model_name == "Vamsi/T5_Paraphrase_Paws":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif model_name == "facebook/bart-base":
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    elif model_name == "t5-base":
        tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    elif model_name == "tuner007/pegasus_paraphrase":
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
    elif model_name == "eugenesiow/bart-paraphrase":
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    else:
        raise Exception("We don't have that model")

    return tokenizer, model


# Estimates the number of tokens to set maximum and minimum lengths parameters
def estimate_tokens_for_words(text, tokenizer):
    """
    This function estimates the number of tokens in a given text using a tokenizer.
    It tokenizes the input text and returns the length of the tokens.
    """
    tokens = tokenizer.tokenize(text)
    return len(tokens)


def paraphrase(phrases, approaches, models, param_grid_transformers, param_grid_llms, output_dir, evaluator):
    """
    This function paraphrases a list of phrases using different approaches and models.
    It generates all possible parameter combinations, paraphrases each phrase, evaluates the results,
    and stores them in a CSV file.

    Parameters:
    - phrases (list): A list of phrases to be paraphrased.
    - approaches (list): A list of approaches to use for paraphrasing.
    - models (list): A list of models to use for paraphrasing.
    - param_grid (dict): A dictionary of parameters to use for paraphrasing.
    - output_dir (str): The directory where the results will be saved.
    - evaluator (object): An object used to evaluate the paraphrased phrases.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"{output_dir}/{timestamp}/", exist_ok=True)
    results = []
    for approach in approaches:
        if approach == "parrot":
            # Init models (make sure you init ONLY once if you integrate this to your code)
            parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
            for phrase in phrases:
                paraphrased_phrases = []
                try:
                    # This function uses the parrot library to paraphrase an input phrase
                    # The library is for conversational interfaces in natural language understanding (nlu)
                    outputs = parrot.augment(
                        input_phrase=phrase,  # The input phrase to be paraphrased
                        use_gpu=False,  # Whether to use GPU for processing
                        diversity_ranker="levenshtein",  # The metric to measure diversity between phrases
                        do_diverse=True,  # Whether to generate diverse paraphrases
                        max_return_phrases=10,  # The maximum number of paraphrased phrases to return
                        max_length=32,  # The maximum length of each paraphrased phrase
                        adequacy_threshold=0.75,  # The minimum threshold for preserving the original meaning
                        fluency_threshold=0.75,
                    )  # The minimum threshold for fluency in the generated phrases
                    # Parse paraphrased sentences
                    for output in outputs:
                        decoded_output = tokenizer.decode(
                            output,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                        paraphrased_phrase = {
                            "phrase": decoded_output,
                            "approach": approach,
                            "model": model_name,
                        }
                        paraphrased_phrases.append(paraphrased_phrase)
                    # Evaluate paraphrases
                    eval_results = evaluator.evaluate_paraphrases(phrase, paraphrased_phrases)
                    # Store results
                    result = {
                        "phrase": phrase,
                        "model": model,
                        **params,
                        **eval_results["aggregate_metrics"],
                        "success": True,
                    }
                except Exception as e:
                    print(f"Error: {str(e)}")
                    result = {
                        "phrase": phrase,
                        "model": model,
                        **params,
                        "success": False,
                        "error": str(e),
                    }
                results.append(result)
        elif approach == "transformers":
            # Initialize Transformers paraphrasers
            paraphrasers = [model_name for model_name in models if model_name in ['Vamsi/T5_Paraphrase_Paws', 'facebook/bart-base', 't5-base', 'tuner007/pegasus_paraphrase']]
            # Generate all parameter combinations
            param_combinations = [dict(zip(param_grid_transformers.keys(), v)) for v in product(*param_grid_transformers.values())]
            for phrase in phrases:
                for model_name in paraphrasers:
                    tokenizer, model = paraphrasing_factory(model_name)
                    # Tokenise the input sentence
                    input_ids = tokenizer.encode(phrase, return_tensors="pt")
                    estimated_tokens = estimate_tokens_for_words(phrase, tokenizer)
                    for params in param_combinations:
                        print(f"\nTesting model: {model_name}")
                        print(f"Parameters: {params}")
                        print(f"Phrase: {phrase[:50]}...")
                        paraphrased_phrases = []
                        try:
                            # Generate paraphrased sentence
                            outputs = model.generate(
                                input_ids=input_ids,  # The input sequence to the model
                                max_length=estimated_tokens
                                * 3,  # The maximum length of the sequence to be generated
                                min_length=estimated_tokens,  # The minimum length of the generated sequence
                                do_sample=True,  # Whether or not to use sampling to include randomness in the generated text
                                temperature=params[
                                    "temperature"
                                ],  # Controls randomness (higher = more random). Range: 0.0 to 1.0
                                top_k=120,  # The number of highest probability vocabulary tokens to keep for top-k filtering
                                top_p=params[
                                    "top_p"
                                ],  # If set to < 1, only the most likely tokens with probabilities that add up to top_p will be kept for generation
                                repetition_penalty=params[
                                    "repetition_penalty"
                                ],  # Penalizes repetition in generated text. Values > 1.0 penalize more
                                length_penalty=1.5,  # Encourages longer/shorter sequences. Values > 1.0 encourage longer sequences
                                no_repeat_ngram_size=2,  # Prevents repetition of n-grams. Set to 0 to disable
                                early_stopping=True,  # Whether to stop the beam search when at least one sentence is finished per batch or not
                                num_return_sequences=5,  # The number of independently computed returned sequences
                                bad_words_ids=None,  # list of token IDs that should not be generated. Set to None to disable
                                num_beams=10,  # Number of beams for beam search. Higher values increase the number of possible sequences
                            )
                            # Parse paraphrased sentences
                            for output in outputs:
                                decoded_output = tokenizer.decode(
                                    output,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True,
                                )
                                paraphrased_phrase = {
                                    "phrase": decoded_output,
                                    "approach": approach,
                                    "model": model_name,
                                }
                                paraphrased_phrases.append(paraphrased_phrase)
                            # Evaluate paraphrases
                            eval_results = evaluator.evaluate_paraphrases(phrase, paraphrased_phrases)
                            # Store results
                            result = {
                                "phrase": phrase,
                                "model": model_name,
                                **params,
                                **eval_results["aggregate_metrics"],
                                "success": True,
                            }
                        except Exception as e:
                            print(f"Error: {str(e)}")
                            result = {
                                "phrase": phrase,
                                "model": model_name,
                                **params,
                                "success": False,
                                "error": str(e),
                            }
                        results.append(result)
        elif approach == "llms":
            # Initialize LLM paraphrasers
            paraphrasers = {
                model_name: LLMParaphraser(model_name) 
                for model_name in models if model_name in 
                ['claude', 'openai', 'gemini', 'llama', 'deepseek']
            }
            # Generate all parameter combinations
            param_combinations = [dict(zip(param_grid_llms.keys(), v)) for v in product(*param_grid_llms.values())]            
            for phrase in phrases:
                for model_name, paraphraser in paraphrasers.items():
                    for params in param_combinations:
                        print(f"\nTesting model: {model_name}")
                        print(f"Parameters: {params}")
                        print(f"Phrase: {phrase[:50]}...")
                        paraphrased_phrases = []
                        try:
                            # Generate paraphrases using LLM
                            paraphrased_phrases = paraphraser.paraphrase(phrase, 10, params["temperature"])
                        
                            # Evaluate paraphrases
                            if paraphrased_phrases:
                                eval_results = evaluator.evaluate_paraphrases(phrase, paraphrased_phrases)
                            
                                # Store results
                                result = {
                                    'phrase': phrase,
                                    'model': model_name,
                                    **params,
                                    'repetition_penalty': None,
                                    'top_p': None,
                                    **eval_results['aggregate_metrics'],
                                    'success': True
                                }
                            else:
                                result = {
                                    'phrase': phrase,
                                    'model': model_name,
                                    **params,
                                    'repetition_penalty': -1.0,
                                    'top_p': -1.0,
                                    'success': False,
                                    'error': 'No paraphrases generated'
                                }
                            
                            results.append(result)
                        
                        except Exception as e:
                            print(f"Error with {model_name}: {str(e)}")
                            results.append({
                                'phrase': phrase,
                                'model': model_name,
                                **params,
                                'repetition_penalty': None,
                                'top_p': None,
                                'success': False,
                                'error': str(e)
                            })
        else:
            raise Exception("We don't have that approach")
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save results
    df.to_csv(f"{output_dir}/{timestamp}/results.csv", index=False)

    return df, timestamp


def run_experiments():
    """
    This function runs the paraphrasing experiments using the provided parameters and models.
    It initializes the evaluator, calls the paraphrase function, and plots the results.
    """
    evaluator = ParaphraseEvaluator()
    # Run experiments
    results_df, timestamp = paraphrase(
        phrases=test_phrases,
        approaches=approaches,
        models=models,
        param_grid_transformers=param_grid_transformers,
        param_grid_llms=param_grid_llms,
        output_dir="paraphrasing_results",
        evaluator=evaluator,
    )
    # Plot results
    summary = evaluator.plot_results(results_df, f"paraphrasing_results/{timestamp}/")
    # Print summary
    print("\nSummary Statistics:")
    print(summary)


if __name__ == "__main__":
    run_experiments()
