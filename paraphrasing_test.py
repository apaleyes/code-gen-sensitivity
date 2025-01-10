from parrot import Parrot
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize

import torch
import warnings
import os
import nltk



warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nltk.download("punkt")

phrases = [
    "Write Python code for addition, subtraction, division, multiplication, and other similar operations, all a part of one class",
    "Given the business logic code below, implement Flask backend. Do not include example of usage or the business logic, do not repeat any code from this prompt. Only write the Flask API code."
]

approaches = ["parrot", "transformers"]


def paraphrase(phrase, approach):
    paraphrased_phrases = []
    if approach == "parrot":
        # init models (make sure you init ONLY once if you integrate this to your code)
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        
        # this function uses the parrot library to paraphrase an input phrase
        # the library is for conversational interfaces in natural language understanding (nlu)
        # diversity_ranker is the distance metric to measure diversity between phrases
        # do_diverse inserts variety to the generated text
        # max_return_phrases is the maximum number of returned phrases
        # max_length of the returned phrases
        # the pre-trained model is trained on text samples of maximum length of 32.
        # adequacy_threshold restricts how the meaning is preserved
        # fluency_thershold how fluent are the phrases in english
        # the number in each generated phrase is the diversity score
        # higher diversity score means the resulting sentence is more diverse than the input
        paraphrased_phrases = parrot.augment(input_phrase=phrase,
                                   use_gpu=False,
                                   diversity_ranker="levenshtein",
                                   do_diverse=True, 
                                   max_return_phrases = 10, 
                                   max_length=32,
                                   adequacy_threshold = 0.75, 
                                   fluency_threshold = 0.75)
    if approach == "transformers":
        tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        sentence = "paraphrase: " + phrase + "</s>"
        encoding = tokenizer.encode_plus(sentence, padding=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

        outputs = model.generate(input_ids=input_ids, max_length=256, do_sample=True, top_k=120, 
                                 top_p=0.95, early_stopping=True, num_return_sequences=5    )
        for output in outputs:
            paraphrased_phrase = tokenizer.decode(output, skip_special_tokens=True, 
                                                  clean_up_tokenization_spaces=True)
            paraphrased_phrases.append(paraphrased_phrase)
    return paraphrased_phrases


for phrase in phrases:
    for approach in approaches:
        paraphrased_phrases = paraphrase(phrase, approach)
        if paraphrased_phrases is not None:
            print("**** original ****")
            print(phrase)
            print("**** paraphrased phrases using " + approach + " ****")
            for paraphrased_phrase in paraphrased_phrases:
                print(paraphrased_phrase) 
        else:
            print("**** none paraphrased phrases using " + approach + " ****")
