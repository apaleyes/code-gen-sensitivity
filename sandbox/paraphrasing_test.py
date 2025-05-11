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
            #-'Write code to create a database schema for an online bicycle shop that sells bicycles as well as their spare parts and accessories. Use SQAlchemy library to communicate with the database. The code should cover opening and closing of a new connection to the database, creation of necessary tables and their relations, common operations on items in the tables such as addition, deletion, updates, filtered selection and search.',
            #'Write a Calculator class. It shall contain common arithmetic operations, such as addition or multiplication, but also more advanced operations, such as logarithm (of variable bases), factorial, trigonometry, roots, exponents.', 
            'Implement a REST API for a web application that implements a personal todo list using Flask. The API should allow a user to create, update and delete whole lists as well as individual items. It should also give users an idea of their progress, and give reminders of tasks due or overdue. You can assume the database layer was already implemented separately.\n'
            #'Implement a script that generates accounting reports for a medium size retail business. It should be customizable to cover variable periods of time (month, quarter, year). You can assume data for the report is coming from one of the common accounting systems, such as QuickBooks or Sage. These reports are intended both for internal consumption (such as verification by accountants or management) and for external use (such as submission to tax authority).', 
            #'Implement a backend engine that tracks user interactions across a website, in an appropriate and efficient data structure. Interactions can be of multiple types, with potential new ones to be added later, which the system must be able to accommodate without major changes. Interactions include likes, ratings, uploads, settings changes. The system should adhere to privacy regulations, and not store any illegal information, and use hashing and anonymisation where possible without losing functionality.\n', 
            #'Write a workout tracking app that can import from several fitness apps including Strava, Polar Flow, MyFitnessPal to put together a dashboard of workouts and nutrition information. The user should be able to customise the views by sport as well as by overall mileage or other stats that transfer across sports such as heart rate or total time. Heart rate data should be aggregated to display the total amount of time at different intensity zones. The user should be able to see a graph of their overall progress by pace or by mileage using a dropdown to switch between them.', 
            #'Implement a simulation of multiple lifts operating in a skyscraper that is used as an office building. The number of lifts should be a tunable parameter, as well as the number of floors in the building. It should also be possible to run the simulation for different crowd sizes. The simulation should clearly display expected behavior, e.g. lifts being busier during the morning and evening hours, used sparingly during the day and barely used at night.', 
            #'Implement a controller system for a network of automated teller machines (ATM). As ATMs operate in a highly sensitive financial setting, the system should have extra measures in place to prevent problems with distributed operations, such as failed transactions, concurrent operations, race conditions. Additionally it should have protection measures in place to prevent fraud.', 
            #'Write code to evaluate a chess position. It can accept chess position as an input in any convenient format, and output evaluation to clearly indicate which side (white or black) has an advantage, and numerically measure this advantage. Adhere to common values of pieces (pawn - 1, knight and bishop - 3, rook - 5, queen - 9), as well as other useful metrics, e.g. number of controlled squares, number of active moves, check and mate possibilities, and so on.', 
            #'The game of Blokus Duo has two players placing pieces in order on a board. The pieces are all the possible combinations of 1x1 square blocks joined together, up to 5 pieces, but cancelling out any symmetries, as the pieces can be flipped and rotated by the players. There is 1 piece that has 1 square (simple 1x1), 1 piece that has 2 squares (simple 1x2), 2 pieces that have 3 squares (1x3 rectangle, and a corner), etc. Write code that finds the number of possible n-square pieces, and lists the possibilities.', 
            #'LEGO claims that 6 standard 2X4 bricks can be connected in 915,103,765 combinations. Write code that verifies this claim, and design a heuristic to decide which combinations are trivial, and which are complicated, giving each a class rating.', 
            #'Write a game of snake like that on the Nokia phones in 1997 that can be played on a laptop while a software update runs. The user should be able to maneuver the snake with the arrow keys. If the snake’s head runs into any part of its body or edge of the game boundary, the game is over. The snake grows by moving around the screen grid and finding “food” in different blocks, getting longer as it eats.', 
            #'Implement gradient ascent algorithm using pure Python. Note that unlike the traditional “gradient descent” algorithm, gradient ascent looks for the global maximum of a function. Do not use any advanced mathematical packages, such as NumPy or SciPy. The algorithm implementation can make reasonable assumptions about smoothness of the function being optimised, but should be able to handle functions that have multiple local maxima.', 
            #'Implement a deep neural network using pure Python. Do not use any deep learning frameworks such as Tensorflow or PyTorch. The network should support fully connected layers, activations, and allow for forwards and backwards passes. It should also allow for a variable number of hidden layers, input and output layers.  Include an example of how this network can be trained for regression and classification problems.\n', 
            #'Given a set of 2n coins, each biased with a different p(heads) value, pick a set of coins that gives the highest probability that when tossed, the number of heads will be n. Note that this is not necessarily the same as expected value. Pick the sets using different methods, and compare their computational complexities.', 
            #'Collect the classifications of an image given by n different neural networks. Knowing the accuracy of each of these networks’ predictions, design an ensemble method that will output the most likely classification given the individual guesses. Also state the confidence level of your guess.', 
            #'Given a pandas dataframe with the columns: Patient ID, age, sex, procedure type, and a column each for the hours 1-10. Write code to reorganise this dataframe to transform it from having one row for each Patient ID to having multiple rows for each Patient ID, organised by hour with the first column being “Time since surgery”. Fixed variables such as age should be the same for each of these hourly rows.', 
            #'Write a sorting function that sorts strings according to underlying ordering. So 8 9 10 should be sorted 8 9 10, not alphabetically 10 8 9. Also take into consideration cases like worded Nine Ten Eleven, or months 31 January - 1 February, etc. It’s one function, and the methods used have to be decided automatically. Default back to alphabetical order if no pattern is found. Partial matching patterns should also be used (eg. iPhone 9, iPhone X, iPhone 11).'
            ]

        # Temperature is the only parameter that influences the BLEU score        
        self.param_grid = {
            #"transformers": {
            #    "temperature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            #    "top_p": [0.7, 0.8, 0.9],
            #    "top_k": [5, 25, 50],
            #    "repetition_penalty": [0.5, 1.0, 1.5, 2.0]
            #},
            "llms": {
                "temperature": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "diversity_rate": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                #"top_p": [0.7, 0.8, 0.9],
                #"top_k": [5, 25, 50],
                #"repetition_penalty": [0.5, 1.0, 1.5, 2.0]
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
        #selected_approaches=["transformers", "llms"],
        selected_approaches=["llms"],
        #selected_models={"transformers": ["tuner007/pegasus_paraphrase"], "llms": ["gemini"]},
        selected_models={"llms": ["gemini"]},
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
