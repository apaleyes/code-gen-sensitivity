import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from models.model_caller import ModelCaller
from models import get_model
from typing import List, Dict

class LLMParaphraser:
    """Class to handle paraphrasing using different LLM models"""
    
    def __init__(self, model_name: str):
        """
        Initialize the LLM paraphraser
        
        Args:
            model_name: Name of the LLM model to use ('claude', 'openai', 'gemini', 'llama', 'deepseek')
        """
        self.model_name = model_name
        self.model = self._initialize_model()
        self.model_caller = ModelCaller(
            model=self.model,
            n_retries=3,
            prompt_transform=self._transform_prompt
        )
        
        # Base prompt template for paraphrasing
        self.base_prompt = """
        You are a paraphrasing assistant. Your task is to paraphrase the given text according to the diversity rate. The diversity rate is the percentage of words you must change in the original text. If the percentage is 0, the original sentence must be kept. If the percentage is 100, all the words must be changed.
        
        Requirements:
        - Maintain the exact same meaning
        - The paraphrases must be semantically similar to the input text
        - Be clear and natural
        - Do not add or remove information
        - Change the words according the diversity rate parameter
        - Ignore your repetition penalty parameter if needed
        - When the rate is between 0 and 50, change words but not the sentence structure
        - When the rate is between 50 and 100, change words and the sentence structure
        
        Text to paraphrase:
        "{text}"

        Diversity rate: "{diversity_rate}"
        
        Generate {num_variations} different paraphrased versions. 
        
        Format your response as a Python list of strings, one paraphrase per string.
        Example format:
        [
            "First paraphrase here",
            "Second paraphrase here",
            "Third paraphrase here"
        ]
        """
    
    def _initialize_model(self):
        """Initialize the specified LLM model"""
        return get_model(self.model_name.lower())
    
    def _transform_prompt(self, text: str) -> str:
        """Transform the input text into a proper prompt"""
        return text
    
    def paraphrase(self, text: str, rule: str, num_variations: int = 5, diversity_rate: int = 100, temperature: float=1.0, top_p: float=0.95, top_k: int=120, frequency_penalty: float=0.0) -> List[Dict]:
        """
        Generate paraphrased versions of the input text
        
        Args:
            text: Text to paraphrase
            num_variations: Number of paraphrased versions to generate
            diversity_percentage: Percentage of words to change in the original sentence
            
        Returns:
            List of dictionaries containing paraphrased versions and metadata
        """
        prompt = self.base_prompt.format(text=text, num_variations=num_variations, rule=rule, diversity_rate=diversity_rate)
        self.model.temperature = temperature
        self.model.top_p = top_p
        self.model.top_k = top_k
        self.model.frequency_penalty = frequency_penalty
        try:
            # Get response from the model using ModelCaller
            response = self.model_caller.get_code(prompt)
            
            # Since we asked for Python list format, we can safely eval it
            # The response should be a valid Python list of strings
            try:
                paraphrases = eval(response)
                if not isinstance(paraphrases, list):
                    raise ValueError("Response is not a list")
            except:
                # If eval fails, try simple line splitting as fallback
                paraphrases = [p.strip() for p in response.split('\n') 
                             if p.strip() and not p.startswith('[') and not p.endswith(']')]
            
            # Format the results
            results = []
            for i, paraphrase in enumerate(paraphrases[:num_variations], 1):
                if isinstance(paraphrase, str):  # Ensure we only include strings
                    results.append({
                        'phrase': paraphrase,
                        'approach': 'llms',
                        'model': self.model_name,
                        'paraphrase_id': i
                    })
            
            return results
            
        except Exception as e:
            print(f"Error with {self.model_name}: {str(e)}")
            return []