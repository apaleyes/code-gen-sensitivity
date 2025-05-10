import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple
import pandas as pd
from paraphrasing_evaluation import ParaphraseEvaluator
from paraphrasing_approaches import ParrotParaphraser, TransformerParaphraser, LLMParaphraserWrapper
from paraphrasing_datasource import DataSource, TestPhrasesDataSource, LeetCodeDataSource, TasksDataSetDataSource, CSVDataSource
from dotenv import load_dotenv

class ParaphrasingExperiment:
    def __init__(self):
        load_dotenv()
        warnings.filterwarnings("ignore")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        
        self.default_test_phrases = [
            #"Write a Calculator class. It shall contain common arithmetic operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry, roots, exponents."
            #"Implement a REST API for a web application that implements a personal todo list using Flask. The API should allow a user to create, update and delete whole lists as well as individual items. It should also give users an idea of their progress, and give reminders of tasks due or overdue. You can assume the database layer was already implemented separately.",
            #"Write code to create a database schema for an online bicycle shop that sells bicycles as well as their spare parts and accessories. Use SQAlchemy library to communicate with the database. The code should cover opening and closing of a new connection to the database, creation of necessary tables and their relations, common operations on items in the tables such as addition, deletion, updates, filtered selection and search.",
            "Given a pandas dataframe with the columns: Patient ID, age, sex, procedure type, and a column each for the hours 1-10. Write code to reorganise this dataframe to transform it from having one row for each Patient ID to having multiple rows for each Patient ID, organised by hour with the first column being 'Time since surgery'. Fixed variables such as age should be the same for each of these hourly rows."
            ]

        # Temperature is the only parameter that influences the BLEU score        
        self.param_grid = {
            "transformers": {
                "temperature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                #"top_p": [0.9, 0.99, 0.999],
                #"top_k": [10, 15, 20]
                #"repetition_penalty": [0.5, 1.0, 1.5, 2.0]
            },
            "llms": {
                "temperature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
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
            file_path = kwargs.get('file_path', 'sandbox/oldleetcode.json')
            return LeetCodeDataSource(file_path)
        elif source_type == "leetcode_new":
            file_path = kwargs.get('file_path', 'sandbox/newleetcode.json')
            return LeetCodeDataSource(file_path)
        elif source_type == "tasks_dataset":
            file_path = kwargs.get('file_path', 'sandbox/ourdataset.json')
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
        no_paraphrases = []
        not_low = []
        not_moderate = []
        not_high = []
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

                            if len(paraphrased_phrases) == 0:
                                print(f"No paraphrased phrases found for {phrase}")
                                no_paraphrases.append(phrase)
                                continue
                            
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
                paraphrases_df.to_csv(f"{results_dir}/paraphrases_{data_source_type}.csv", index=False)
                print(f"All paraphrases saved to {results_dir}/paraphrases_{data_source_type}.csv")
        
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
                plt.savefig(f"{results_dir}/semantic_similarity_vs_bleu_{data_source_type}.png")
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
                plt.savefig(f"{results_dir}/bert_score_vs_sacre_bleu_{data_source_type}.png")
                plt.close()

        for phrase_data in data_source.get_phrases():
            phrase = phrase_data['text']
            paraphrases_df = pd.DataFrame(paraphrases)
            paraphrased_phrase = paraphrases_df[paraphrases_df['original_phrase'] == phrase]
            if len(paraphrased_phrase) == 0:
                no_paraphrases.append(phrase)
            else:
                low = 0
                moderate = 0
                high = 0
                for index, result in paraphrased_phrase.iterrows():
                    if result['sacre_bleu'] < 0.3333333333333333:
                        low = low + 1
                    elif result['sacre_bleu'] < 0.6666666666666666:
                        moderate = moderate + 1
                    else:
                        high = high + 1

                    if low > 0 and moderate > 0 and high > 0:
                        break
                
                if low == 0:
                    not_low.append(phrase)
                if moderate == 0:
                    not_moderate.append(phrase)
                if high == 0:
                    not_high.append(phrase)
        
        print(f"No paraphrases: {len(no_paraphrases)}")
        print(f"Not low: {len(not_low)}")
        print(f"Not moderate: {len(not_moderate)}")
        print(f"Not high: {len(not_high)}")
        return paraphrases, not_low, not_moderate, not_high

if __name__ == "__main__":
    experiment = ParaphrasingExperiment()
    
    # Example using our dataset
    dataset, not_low_our, not_moderate_our, not_high_our = experiment.run_experiments(
        selected_approaches=["transformers", "llms"],
        selected_models={"transformers": ["tuner007/pegasus_paraphrase"], "llms": ["gemini"]},
        data_source_type="tasks_dataset"
    )

    # Example using old leetcode dataset
    #dataset, not_low_leetcode, not_moderate_leetcode, not_high_leetcode = experiment.run_experiments(
    #    selected_approaches=["transformers", "llms"],
    #    selected_models={"transformers": ["tuner007/pegasus_paraphrase"], "llms": ["gemini"]},
    #    data_source_type="leetcode"
    #)

    # Example using new leetcode dataset
    #dataset, not_low_leetcode_new, not_moderate_leetcode_new, not_high_leetcode_new = experiment.run_experiments(
    #    selected_approaches=["transformers", "llms"],
    #    selected_models={"transformers": ["tuner007/pegasus_paraphrase"], "llms": ["gemini"]},
    #    data_source_type="leetcode_new"
    #)

    # Print missing paraphrases
    print("Our dataset:")
    print(f"Not low")
    print(not_low_our)
    print(f"Not moderate")
    print(not_moderate_our)
    print(f"Not high")
    print(not_high_our)
    #print("Old leetcode dataset:")
    #print(f"Not low")
    #print(not_low_leetcode)
    #print(f"Not moderate")
    #print(not_moderate_leetcode)
    #print(f"Not high")
    #print(not_high_leetcode)
    #print("New leetcode dataset:")
    #print(f"Not low")
    #print(not_low_leetcode_new)
    #print(f"Not moderate")
    #print(not_moderate_leetcode_new)
    #print(f"Not high")
    #print(not_high_leetcode_new)
    # Print or process the dataset
    #for entry in dataset:
    #    print(entry)
