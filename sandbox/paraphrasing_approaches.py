from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
from parrot import Parrot
from llm_paraphraser import LLMParaphraser
from llm_paraphraser_personas import LLMParaphraserPersonas

class BaseParaphraser(ABC):
    def __init__(self, approach_name: str):
        self.approach_name = approach_name
    
    @abstractmethod
    def paraphrase(self, phrase: str, num_variations: int = 5, **kwargs) -> List[Dict]:
        """Generate paraphrased versions of the input text"""
        pass

    def format_result(self, phrase: str, model_name: str, success: bool, metrics: Optional[Dict] = None, error: Optional[str] = None, **kwargs) -> Dict:
        """Helper to format results consistently"""
        result = {
            "phrase": phrase,
            "model": model_name,
            "approach": self.approach_name,
            "success": success,
            **kwargs
        }
        if metrics:
            result.update(metrics)
        if error:
            result["error"] = error
        return result

class ParrotParaphraser(BaseParaphraser):
    def __init__(self):
        super().__init__("parrot")
        self._parrot = None  # Lazy load Parrot

    @property
    def parrot(self):
        if self._parrot is None:
            self._parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        return self._parrot
    
    def paraphrase(self, phrase: str, num_variations: int = 5, **kwargs) -> List[Dict]:
        try:
            outputs = self.parrot.augment(
                input_phrase=phrase,
                use_gpu=torch.cuda.is_available(),
                diversity_ranker="levenshtein",
                do_diverse=True,
                max_return_phrases=num_variations,
                max_length=32,
                adequacy_threshold=0.75,
                fluency_threshold=0.75,
            )
            
            return [{
                "phrase": output,
                "approach": self.approach_name,
                "model": "parrot"
            } for output in outputs]
        except Exception as e:
            print(f"Parrot error: {str(e)}")
            return []

class TransformerParaphraser(BaseParaphraser):
    def __init__(self):
        super().__init__("transformers")
        self._supported_models = None
        self.models = {}
        self.tokenizers = {}

    @property
    def supported_models(self):
        if self._supported_models is None:
            # Import transformers only when needed
            from transformers import (
                BartForConditionalGeneration, BartTokenizer,
                T5ForConditionalGeneration, T5Tokenizer,
                PegasusForConditionalGeneration, PegasusTokenizer
            )
            
            self._supported_models = {
                "facebook/bart-base": (BartTokenizer, BartForConditionalGeneration),
                "t5-base": (T5Tokenizer, T5ForConditionalGeneration),
                "tuner007/pegasus_paraphrase": (PegasusTokenizer, PegasusForConditionalGeneration),
                "eugenesiow/bart-paraphrase": (BartTokenizer, BartForConditionalGeneration)
            }
        return self._supported_models

    def load_model(self, model_name: str):
        """Load and cache model and tokenizer"""
        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}")
            
        if model_name not in self.models:
            tokenizer_class, model_class = self.supported_models[model_name]
            self.tokenizers[model_name] = tokenizer_class.from_pretrained(model_name)
            self.models[model_name] = model_class.from_pretrained(model_name)
            
            if torch.cuda.is_available():
                self.models[model_name].to('cuda')
        
        return self.tokenizers[model_name], self.models[model_name]

    def estimate_tokens(self, text: str, tokenizer) -> int:
        """Estimate the number of tokens in the text"""
        tokens = tokenizer.tokenize(text)
        return len(tokens)

    def paraphrase(self, phrase: str, num_variations: int = 5, **kwargs) -> List[Dict]:
        model_name = kwargs.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for transformer approach")

        try:
            tokenizer, model = self.load_model(model_name)

            paraphrases = [""] * num_variations
            sentences = phrase.split(".")
            sentences = [sentence for sentence in sentences if sentence.strip()]
            for sentence in sentences:
                # Prepare input
                input_ids = tokenizer.encode(sentence, return_tensors="pt")
                if torch.cuda.is_available():
                    input_ids = input_ids.to('cuda')
                
                estimated_tokens = self.estimate_tokens(sentence, tokenizer)
            
                # Generate paraphrases
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=estimated_tokens * 3,
                    min_length=estimated_tokens,
                    do_sample=True,
                    temperature=kwargs.get("temperature", 1.0),
                    top_k=120,
                    top_p=kwargs.get("top_p", 0.95),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.5),
                    length_penalty=1.5,
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                    num_return_sequences=num_variations,
                    num_beams=10,
                )
                index = 0
                for output in outputs:
                    paraphrased_sentence = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True),
                    paraphrases[index] = paraphrases[index] + paraphrased_sentence[0] + " "
                    index = index + 1

            return [{
                "phrase": phrased_phrase,
                "approach": self.approach_name,
                "model": model_name
            } for phrased_phrase in paraphrases]
        except Exception as e:
            print(f"Transformer error: {str(e)}")
            return []

class LLMParaphraserWrapper(BaseParaphraser):
    def __init__(self):
        super().__init__("llms")
        self.paraphrasers = {}
        
    def get_paraphraser(self, model_name: str) -> LLMParaphraser:
        """Get or create LLM paraphraser instance"""
        if model_name not in self.paraphrasers:
            self.paraphrasers[model_name] = LLMParaphraser(model_name)
        return self.paraphrasers[model_name]

    def paraphrase(self, phrase: str, num_variations: int = 5, **kwargs) -> List[Dict]:
        model_name = kwargs.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for LLM approach")
            
        try:
            paraphraser = self.get_paraphraser(model_name)
            return paraphraser.paraphrase(
                phrase, 
                num_variations, 
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 120),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0)
            )
        except Exception as e:
            print(f"LLM error: {str(e)}")
            return []

class LLMParaphraserPersonasWrapper(BaseParaphraser):
    def __init__(self):
        super().__init__("llms")
        self.paraphrasers = {}
        
    def get_paraphraser(self, model_name: str) -> LLMParaphraserPersonas:
        """Get or create LLM paraphraser instance"""
        if model_name not in self.paraphrasers:
            self.paraphrasers[model_name] = LLMParaphraserPersonas(model_name)
        return self.paraphrasers[model_name]

    def paraphrase(self, phrase: str, num_variations: int = 5, **kwargs) -> List[Dict]:
        model_name = kwargs.get("model_name")
        if not model_name:
            raise ValueError("model_name is required for LLM approach")
            
        try:
            paraphraser = self.get_paraphraser(model_name)
            return paraphraser.paraphrase(
                phrase, 
                num_variations, 
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 120),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0)
            )
        except Exception as e:
            print(f"LLM error: {str(e)}")
            return [] 