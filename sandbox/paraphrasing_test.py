from parrot import Parrot
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

import torch
import warnings
import os
import nltk


from paraphraser_evaluation import ParaphraseEvaluator


warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download("punkt")

phrases = [
    "Write Python code for addition, subtraction, division, multiplication, and other similar operations, all a part of one class",
    "Given the business logic code below, implement Flask backend. Do not include example of usage or the business logic, do not repeat any code from this prompt. Only write the Flask API code."
]

approaches = ["transformers"]
models = ["Vamsi/T5_Paraphrase_Paws", "facebook/bart-base", "t5-base", "tuner007/pegasus_paraphrase"]


# creates tokenizers and models for paraphrasing
def paraphrasing_factory(model_name):
    if model_name == "Vamsi/T5_Paraphrase_Paws":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if model_name == "facebook/bart-base":
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
    if model_name == "t5-base":
        tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    if model_name == "tuner007/pegasus_paraphrase":
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


# estimates the number of tokens to set maximum and minimum lengths parameters
def estimate_tokens_for_words(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def paraphrase(phrase, approach):
    if approach == "parrot":
        paraphrased_phrases = []
        # init models (make sure you init ONLY once if you integrate this to your code)
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        
        # this function uses the parrot library to paraphrase an input phrase
        # the library is for conversational interfaces in natural language understanding (nlu)
        outputs = parrot.augment(input_phrase=phrase,  # The input phrase to be paraphrased
                                   use_gpu=False,  # Whether to use GPU for processing
                                   diversity_ranker="levenshtein",  # The metric to measure diversity between phrases
                                   do_diverse=True,  # Whether to generate diverse paraphrases
                                   max_return_phrases=10,  # The maximum number of paraphrased phrases to return
                                   max_length=32,  # The maximum length of each paraphrased phrase
                                   adequacy_threshold=0.75,  # The minimum threshold for preserving the original meaning
                                   fluency_threshold=0.75)  # The minimum threshold for fluency in the generated phrases
        for output in outputs:
            paraphrased_phrase = {"phrase": output, "approach": approach}

    if approach == "transformers":
        for model_name in models:
            paraphrased_phrases = []
            tokenizer, model = paraphrasing_factory(model_name)            
            #sentence = "paraphrase: " + phrase + "</s>"
            # tokenise the input sentence
            input_ids = tokenizer.encode(phrase, return_tensors='pt')
            estimated_tokens = estimate_tokens_for_words(phrase, tokenizer)
            # generate paraphrased sentence
            outputs = model.generate(
                input_ids=input_ids,  # the input sequence to the model
                max_length=estimated_tokens * 3,  # the maximum length of the sequence to be generated
                min_length=estimated_tokens,  # the minimum length of the generated sequence
                do_sample=True,  # whether or not to use sampling to include randomness in the generated text
                temperature=0.7,  # controls randomness (higher = more random). Range: 0.0 to 1.0
                top_k=120,  # the number of highest probability vocabulary tokens to keep for top-k filtering
                top_p=0.95,  # if set to < 1, only the most likely tokens with probabilities that add up to top_p will be kept for generation
                repetition_penalty=1.2,  # penalizes repetition in generated text. Values > 1.0 penalize more
                length_penalty=1.5,  # encourages longer/shorter sequences. Values > 1.0 encourage longer sequences
                no_repeat_ngram_size=2,  # prevents repetition of n-grams. Set to 0 to disable
                early_stopping=True,  # whether to stop the beam search when at least one sentence is finished per batch or not
                num_return_sequences=5,  # the number of independently computed returned sequences
                bad_words_ids=None,  # list of token IDs that should not be generated. Set to None to disable
                num_beams=10,  # number of beams for beam search. Higher values increase the number of possible sequences
            )

            evaluator = ParaphraseEvaluator()
            for output in outputs:
                decoded_output = tokenizer.decode(output, skip_special_tokens=True, 
                                                      clean_up_tokenization_spaces=True)
                paraphrased_phrase = {"phrase": decoded_output, "approach": approach, "model": model_name}    
                paraphrased_phrases.append(paraphrased_phrase)
            evaluation_results = evaluator.evaluate_paraphrases(phrase, paraphrased_phrases)
            evaluator.print_evaluation_results(evaluation_results)    
    return paraphrased_phrases


for phrase in phrases:
    for approach in approaches:
        paraphrased_phrases = paraphrase(phrase, approach)
