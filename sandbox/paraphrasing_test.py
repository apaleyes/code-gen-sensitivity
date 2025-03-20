import os
import warnings
import json
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple
import pandas as pd
from paraphrasing_evaluation import ParaphraseEvaluator
from paraphrasing_approaches import ParrotParaphraser, TransformerParaphraser, LLMParaphraserWrapper
from paraphrasing_datasource import DataSource, TestPhrasesDataSource, LeetCodeDataSource, CSVDataSource
from dotenv import load_dotenv

class ParaphrasingExperiment:
    def __init__(self):
        load_dotenv()
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.default_test_phrases = [
            "Write Python code for addition, subtraction, division, multiplication, and other similar operations, all a part of one class",
            "Given the business logic code below, implement Flask backend. Do not include example of usage or the business logic, do not repeat any code from this prompt. Only write the Flask API code.",
            "Write a Calculator class. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry."
        ]
        
        self.param_grid = {
            "transformers": {
                "temperature": [0.0, 0.5, 1.0],
                "repetition_penalty": [1.5],
                "top_p": [0.95]
            },
            "llms": {
                "temperature": [0.0],
                "top_p": [0.95],
                "top_k": [40],
                "frequency_penalty": [-2.0]
            }
        }
        self.diversity_ranges = [(0.0, 0.4), (0.4, 0.8), (0.8, 1.0)]
        self.paraphrases_range = 10
        
        self.evaluator = ParaphraseEvaluator()
        self._approaches = {}  # Empty dict for lazy loading

    def get_approach(self, approach_name: str):
        """Lazy load approaches only when needed"""
        if approach_name not in self._approaches:
            if approach_name == "parrot":
                self._approaches[approach_name] = ParrotParaphraser()
            elif approach_name == "transformers":
                self._approaches[approach_name] = TransformerParaphraser()
            elif approach_name == "llms":
                self._approaches[approach_name] = LLMParaphraserWrapper()
            else:
                raise ValueError(f"Unsupported approach: {approach_name}")
        return self._approaches[approach_name]

    def create_data_source(self, source_type: str, **kwargs) -> DataSource:
        """Factory method to create data sources"""
        if source_type == "test_phrases":
            phrases = kwargs.get('phrases', self.default_test_phrases)
            return TestPhrasesDataSource(phrases)
        elif source_type == "leetcode":
            file_path = kwargs.get('file_path', 'sandbox/leetcode-dataset.json')
            return LeetCodeDataSource(file_path)
        elif source_type == "csv":
            file_path = kwargs.get('file_path')
            text_column = kwargs.get('text_column')
            return CSVDataSource(file_path, text_column)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
        
    def dataset_complete(self, dataset: List[Dict]) -> bool:
        for range in dataset:
            if len(range['paraphrases']) < self.paraphrases_range:
                return False
        return True

    def run_experiments(self, 
                       selected_approaches: List[str], 
                       selected_models: List[str], 
                       data_source_type: str = "test_phrases",
                       data_source_kwargs: Dict = None,
                       output_dir: str = "paraphrasing_results") -> List[Tuple[str, List[str], float, float]]:
        """
        Run paraphrasing experiments and generate dataset
        
        Returns:
            List of tuples (original_phrase, [paraphrased_phrases], semantic_similarity, bleu_score)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"{output_dir}/{timestamp}/"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create data source
        data_source_kwargs = data_source_kwargs or {}
        data_source = self.create_data_source(data_source_type, **data_source_kwargs)
        
        dataset = []
        for range_start, range_end in self.diversity_ranges:
            dataset.append({
                "diversity_range": (range_start, range_end),
                "paraphrases_range": self.paraphrases_range,
                "paraphrases": []
            })
        
        for approach_name in selected_approaches:
            try:
                approach = self.get_approach(approach_name)
            except ValueError as e:
                print(f"Warning: {str(e)}")
                continue
                
            param_grid = self.param_grid.get(approach_name, {})
            param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
            
            # If no parameters defined, still run once with defaults
            if not param_combinations:
                param_combinations = [{}]
            
            for phrase_data in data_source.get_phrases():
                phrase = phrase_data['text']
                for model_name in selected_models:
                    for params in param_combinations:
                        if not self.dataset_complete(dataset):
                            print(f"\nTesting {approach_name} with {model_name}")
                            print(f"Parameters: {params}")
                            print(f"Phrase: {phrase[:50]}...")
                            try:
                                paraphrased_phrases = approach.paraphrase(
                                    phrase, 
                                    num_variations=5,
                                    model_name=model_name,
                                    **params
                                )
                            
                                if paraphrased_phrases:
                                    eval_results = self.evaluator.evaluate_paraphrases(phrase, paraphrased_phrases)
                                    individual_results = eval_results['individual_results']
                                    for result in individual_results:
                                        for range in dataset:
                                            if len(range['paraphrases']) < self.paraphrases_range:
                                                if range['diversity_range'][0] <= result['bleu'] <= range['diversity_range'][1]:
                                                    paraphrase = {
                                                        "original_phrase": phrase,
                                                        "paraphrase": result['text'],
                                                        "semantic_similarity": result['semantic_similarity'],
                                                        "bleu": result['bleu']
                                                    }
                                                    range['paraphrases'].append(paraphrase)
                                                    break
                            except Exception as e:
                                print(f"Error: {str(e)}")
                        else:
                            print(f"Dataset complete!")
                            break
        # Save dataset to a file
        with open(f"{results_dir}/paraphrase_dataset.json", 'w') as f:
            json.dump(dataset, f, indent=2)
        
        return dataset

if __name__ == "__main__":
    experiment = ParaphrasingExperiment()
    
    # Example using test phrases
    dataset = experiment.run_experiments(
        selected_approaches=["llms"],
        selected_models=["gemini"],
        data_source_type="test_phrases"
    )
    
    # Print or process the dataset
    for entry in dataset:
        print(entry)
