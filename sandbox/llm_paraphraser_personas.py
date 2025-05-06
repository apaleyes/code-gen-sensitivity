import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from models.model_caller import ModelCaller
from models import get_model
from typing import List, Dict
import json

class LLMParaphraserPersonas:
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

        # Load the personas from file
        with open("personas.json", "r") as file:
            personas = json.load(file)
        
        # Base prompt template for paraphrasing
        for persona in personas:
            self.base_prompt = """
            Write a paraphrase of this prompt as {persona}. Your task is to write a detailed prompt to an LLM for the given example.
            
            Text to write a prompt for:
            "{text}" 
            
            Format your response as a Python string,
            Example format:
            [
                "Prompt here"
            ]
            """
    
    def _initialize_model(self):
        """Initialize the specified LLM model"""
        return get_model(self.model_name.lower())
    
    def _transform_prompt(self, text: str) -> str:
        """Transform the input text into a proper prompt"""
        return text
    
    def paraphrase(self, text: str, rule: str, num_variations: int = 5, temperature: float=1.0, top_p: float=0.95, top_k: int=120, frequency_penalty: float=0.0) -> List[Dict]:
        """
        Generate paraphrased versions of the input text
        
        Args:
            persona: Persona to use for prompt task
            text: Text to use as basis for prompt
            
        Returns:
            List of dictionaries containing paraphrased versions and metadata
        """
        prompt = self.base_prompt.format(text=text, num_variations=num_variations, rule=rule)
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