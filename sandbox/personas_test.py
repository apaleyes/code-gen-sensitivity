import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple
import pandas as pd
from paraphrasing_evaluation import ParaphraseEvaluator
from paraphrasing_approaches import LLMParaphraserPersonasWrapper
from paraphrasing_datasource import DataSource, TestPhrasesDataSource, LeetCodeDataSource, TasksDataSetDataSource, CSVDataSource
from dotenv import load_dotenv

class ParaphrasingExperiment:
    def __init__(self):
        load_dotenv()
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.default_test_phrases = [
            # "Write Python code for addition, subtraction, division, multiplication, and other similar operations, all a part of one class",
            # "Given the business logic code below, implement Flask backend. Do not include example of usage or the business logic, do not repeat any code from this prompt. Only write the Flask API code.",
            "Write a Calculator class. It shall contain common operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry."
        ]

        # Temperature is the only parameter that influences the BLEU score        
        self.param_grid = {
            "llms": {
                "temperature": [1.0],
                #"top_p": [0.9, 0.99, 0.999],
                #"top_k": [10, 15, 20]
                #"frequency_penalty": [-2.0, 0.0, 1.9]
            }
        }
        
        self.evaluator = ParaphraseEvaluator()
        self._approaches = {}  # Empty dict for lazy loading

    def get_approach(self, approach_name: str):
        """Lazy load approaches only when needed"""
        if approach_name not in self._approaches:
            if approach_name == "llms":
                self._approaches[approach_name] = LLMParaphraserPersonasWrapper()
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
         elif source_type == "tasks_dataset":
             file_path = kwargs.get('file_path', 'sandbox/tasks_dataset.json')
             return TasksDataSetDataSource(file_path)
         elif source_type == "csv":
             file_path = kwargs.get('file_path')
             text_column = kwargs.get('text_column')
             return CSVDataSource(file_path, text_column)
         else:
             raise ValueError(f"Unsupported data source type: {source_type}")
        
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
        
        paraphrases = []
        for approach_name in selected_approaches:
            try:
                approach = self.get_approach(approach_name)
            except ValueError as e:
                print(f"Warning: {str(e)}")
                continue
            #rule = 0
            param_grid = self.param_grid.get(approach_name, {})
            param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
            
            # If no parameters defined, still run once with defaults
            if not param_combinations:
                param_combinations = [{}]
            
            for phrase_data in data_source.get_phrases():
                phrase = phrase_data['text']
                for model_name in selected_models[approach_name]:
                    for params in param_combinations:
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
                                    paraphrase = {
                                        "original_phrase": phrase,
                                        "paraphrase": result['text'],
                                        "approach_name": approach_name,
                                        "semantic_similarity": result['semantic_similarity'],
                                        "bleu": result['bleu'],
                                        "bert_score": result['bert_score'],
                                        "sacre_bleu": result['sacre_bleu'],
                                        **params
                                    }
                                    paraphrases.append(paraphrase)
                        except Exception as e:
                            print(f"Error: {str(e)}")
                        
               

                # Save paraphrases to a CSV file
                paraphrases_df = pd.DataFrame(paraphrases)
                paraphrases_df.to_csv(f"{results_dir}/paraphrases.csv", index=False)
                print(f"All paraphrases saved to {results_dir}/paraphrases.csv")
        
                # Plot semantic_similarity vs BLEU
                plt.figure(figsize=(10, 6))
                for approach_name in paraphrases_df['approach_name'].unique():
                    approach_data = paraphrases_df[paraphrases_df['approach_name'] == approach_name]
                    plt.scatter(approach_data['semantic_similarity'], approach_data['bleu'], alpha=0.5, label=approach_name)
                plt.title('Semantic Similarity vs BLEU')
                plt.xlabel('Semantic Similarity')
                plt.ylabel('BLEU Score')
                plt.grid(True)
                plt.legend()
                plt.savefig(f"{results_dir}/semantic_similarity_vs_bleu.png")
                plt.close()

                # Plot BERT score vs Sacre BLEU
                plt.figure(figsize=(10, 6))
                for approach_name in paraphrases_df['approach_name'].unique():
                    approach_data = paraphrases_df[paraphrases_df['approach_name'] == approach_name]
                    plt.scatter(approach_data['bert_score'], approach_data['sacre_bleu'], alpha=0.5, label=approach_name)
                plt.title('BERT Score vs Sacre BLEU')
                plt.xlabel('BERT Score')
                plt.ylabel('Sacre BLEU')
                plt.grid(True)
                plt.legend()
                plt.savefig(f"{results_dir}/bert_score_vs_sacre_bleu.png")
                plt.close()

        return paraphrases

if __name__ == "__main__":
    experiment = ParaphrasingExperiment()
    
    # Example using test phrases
    dataset = experiment.run_experiments(
        selected_approaches=["llms"],
        selected_models={"llms": ["gemini"]},
        data_source_type="test_phrases",
    )
    
    # Print or process the dataset
    #for entry in dataset:
    #    print(entry)