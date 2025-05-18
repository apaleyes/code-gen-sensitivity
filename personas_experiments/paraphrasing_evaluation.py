from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import spacy
import numpy as np
from typing import List, Dict
import torchmetrics
import nltk
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import os
import json
from datetime import datetime
import torch

nltk.download('punkt_tab')


class ParaphraseEvaluator:
    def __init__(self):
        """Initialize the evaluator with necessary models and scorers"""
        # Suppress warnings
        warnings.filterwarnings("ignore")
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK download failed: {e}")
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )

        # Move BERTScore to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_score_metric = torchmetrics.text.BERTScore(
            model_name_or_path="roberta-large"
        ).to(self.device)
        self.sacre_bleu_metric = torchmetrics.text.SacreBLEUScore().to(self.device)

    def evaluate_single_paraphrase(self, original: str, paraphrase: str) -> Dict:
        """
        Evaluate a single paraphrase using multiple metrics
        
        Args:
            original (str): Original text
            paraphrase (str): Paraphrased text
            
        Returns:
            Dict: Dictionary containing various evaluation metrics
        """
        # Tokenize texts
        original_tokens = word_tokenize(original.lower())
        paraphrase_tokens = word_tokenize(paraphrase.lower())

        # Calculate BLEU Score
        # BLEU (Bilingual Evaluation Understudy) is a metric for evaluating a generated sentence to a reference sentence.
        # It is based on the precision of the n-grams (contiguous sequences of n items) in the generated sentence that match the reference sentence.
        # A higher BLEU Score means lower diversity as the generated sentence has more words that appear in the original text.
        # A lower BLEU Score means higher diversity but the meaning can be lost.
        bleu = sentence_bleu([original_tokens], paraphrase_tokens)

        # Calculate SacreBLEU Score
        # SacreBLEU is a metric for evaluating the quality of machine translation outputs.
        # It is based on the BLEU score but with some improvements to make it more accurate and robust.
        # A higher SacreBLEU Score means better translation quality.
        sacre_bleu = self.sacre_bleu_metric([paraphrase], [[original]]).item()

        # Calculate ROUGE Scores
        # ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of text summarization and machine translation algorithms.
        # It measures the overlap between the generated summary and the reference summary based on n-grams.
        # A higher ROUGE Score means lower diversity as the generated sentence overlaps with the original text.
        # A lower ROUGE Score means higher diversity but the meaning can be lost.
        # ROUGE1 - unigram
        # ROUGE2 - bigram
        # ROUGEL - longest matching sequence
        rouge_scores = self.rouge_scorer.score(original, paraphrase)

        # Calculate Semantic Similarity using spaCy
        # Semantic similarity measures the degree of similarity in meaning between two texts.
        # In this case, it's calculated using the cosine similarity between the vector representations of the two texts.
        doc1 = self.nlp(original)
        doc2 = self.nlp(paraphrase)
        semantic_similarity = doc1.similarity(doc2)

        # Calculate Semantic Similarity using BERTScore
        # BERTScore is a metric that measures the similarity between two sentences based on their semantic meaning.
        # It uses a pre-trained BERT model to generate contextualized embeddings for each sentence and then computes the similarity between these embeddings.
        # A higher BERTScore indicates higher semantic similarity between the two sentences.
        bert_score = self.bert_score_metric([paraphrase], [original])
        bert_score_f1 = bert_score['f1'].item()
        
        # Calculate Length Ratio
        # Length ratio measures the proportion of the length of the paraphrased text to the original text.
        # It can indicate if the paraphrased text is more concise or verbose than the original.
        length_ratio = len(paraphrase_tokens) / len(original_tokens)

        # Calculate Lexical Diversity
        # Lexical diversity measures the variety of words used in a text.
        # It's calculated as the ratio of unique words to the total number of words in the paraphrased text.
        # A higher lexical diversity indicates a more diverse use of vocabulary.
        lexical_diversity = len(set(paraphrase_tokens)) / len(paraphrase_tokens)

        return {
            'bleu': bleu,
            'sacre_bleu': sacre_bleu,
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'semantic_similarity': semantic_similarity,
            'bert_score': bert_score_f1,
            'length_ratio': length_ratio,
            'lexical_diversity': lexical_diversity
        }

    def evaluate_readability(self, text: str) -> Dict:
        """
        Calculate readability metrics for the text
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            Dict: Dictionary containing readability metrics
        """
        doc = self.nlp(text)
        
        # Count sentences and words
        sentences = list(doc.sents)
        num_sentences = len(sentences)
        num_words = len([token for token in doc if not token.is_punct])
        
        # Calculate words per sentence
        words_per_sentence = num_words / num_sentences if num_sentences > 0 else 0
        
        # Calculate average word length
        avg_word_length = np.mean([len(token.text) for token in doc if not token.is_punct]) if num_words > 0 else 0
        
        return {
            'num_sentences': num_sentences,
            'num_words': num_words,
            'words_per_sentence': words_per_sentence,
            'avg_word_length': avg_word_length
        }

    def evaluate_grammar(self, text: str) -> Dict:
        """
        Perform basic grammar checking using spaCy
        
        Args:
            text (str): Text to evaluate
            
        Returns:
            Dict: Dictionary containing grammar check results
        """
        doc = self.nlp(text)
        
        # Check for basic subject-verb agreement
        has_subject = False
        has_verb = False
        has_object = False
        
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass']:
                has_subject = True
            if token.pos_ == 'VERB':
                has_verb = True
            if token.dep_ in ['dobj', 'pobj']:
                has_object = True
        
        return {
            'has_subject': has_subject,
            'has_verb': has_verb,
            'has_object': has_object,
            'is_complete': has_subject and has_verb
        }

    def evaluate_paraphrases(self, original: str, paraphrases: List[Dict]) -> Dict:
        """
        Evaluate a list of paraphrases and provide comprehensive metrics
        
        Args:
            original (str): Original text
            paraphrases (List[Dict]): List of dictionaries containing paraphrases
            
        Returns:
            Dict: Dictionary containing evaluation results
        """
        results = []
        
        for idx, paraphrase_dict in enumerate(paraphrases):
            paraphrase_text = paraphrase_dict['phrase']
            approach = paraphrase_dict.get('approach', 'unknown')
            model = paraphrase_dict.get('model', 'unknown')
            
            # Get all metrics
            metrics = self.evaluate_single_paraphrase(original, paraphrase_text)
            #readability = self.evaluate_readability(paraphrase_text)
            #grammar = self.evaluate_grammar(paraphrase_text)

            # Combine all metrics
            result = {
                'paraphrase_id': idx,
                'original': original,
                'text': paraphrase_text,
                'approach': approach,
                'model': model,
                **metrics
                #'readability': readability,
                #'grammar': grammar
            }
            
            results.append(result)
        
        # Calculate aggregate metrics
        avg_metrics = {
            'avg_bleu': np.mean([r['bleu'] for r in results]),
            'avg_rougel': np.mean([r['rougeL'] for r in results]),
            'avg_semantic_similarity': np.mean([r['semantic_similarity'] for r in results]),
            'diversity_between_paraphrases': self.calculate_diversity([r['text'] for r in results])
        }
        
        return {
            'individual_results': results,
            'aggregate_metrics': avg_metrics
        }

    def calculate_diversity(self, paraphrases: List[str]) -> float:
        """
        Calculate diversity between paraphrases using average pairwise BLEU score
        
        Args:
            paraphrases (List[str]): List of paraphrased texts
            
        Returns:
            float: Diversity score (lower score means more diverse paraphrases)
        """
        if len(paraphrases) < 2:
            return 0.0
        
        scores = []
        for i in range(len(paraphrases)):
            for j in range(i + 1, len(paraphrases)):
                ref_tokens = word_tokenize(paraphrases[i].lower())
                hyp_tokens = word_tokenize(paraphrases[j].lower())
                score = sentence_bleu([ref_tokens], hyp_tokens)
                scores.append(score)
        
        return np.mean(scores)

      
    def print_evaluation_results(self, evaluation_results: Dict):
        """
        Print evaluation results in a formatted way
        
        Args:
            evaluation_results (Dict): Dictionary containing evaluation results
        """
        print("\nEvaluation Results:")
        print("=" * 80)
        
        # Print individual results
        for result in evaluation_results['individual_results']:
            print(f"\nParaphrase {result['paraphrase_id']}:")
            print(f"   Original Text: {result['original']}")
            print(f"Paraphrased Text: {result['text']}")
            print(f"Approach: {result['approach']}")
            print(f"Model: {result['model']}")
            print(f"BLEU Score: {result['bleu']:.3f}")
            print(f"ROUGE-L F1: {result['rougel']:.3f}")
            print(f"Semantic Similarity: {result['semantic_similarity']:.3f}")
            print(f"Length Ratio: {result['length_ratio']:.3f}")
            print(f"Lexical Diversity: {result['lexical_diversity']:.3f}")
            print(f"Words per Sentence: {result['readability']['words_per_sentence']:.1f}")
            print(f"Grammar Complete: {'Yes' if result['grammar']['is_complete'] else 'No'}")
        
        # Print aggregate metrics
        print("\nAggregate Metrics:")
        print("-" * 80)
        agg = evaluation_results['aggregate_metrics']
        print(f"Average BLEU Score: {agg['avg_bleu']:.3f}")
        print(f"Average ROUGE-L: {agg['avg_rougel']:.3f}")
        print(f"Average Semantic Similarity: {agg['avg_semantic_similarity']:.3f}")
        print(f"Diversity Between Paraphrases: {agg['diversity_between_paraphrases']:.3f}")


    def plot_results(self, df: pd.DataFrame, output_dir: str = "paraphrasing_results"):
        """
        Create various plots to visualize experimental results
        
        Args:
            df: DataFrame containing experimental results
            output_dir: Directory to save plots
        """
        # Filter successful experiments
        df_success = df[df['success']]

        # Create plots directory
        plots_dir = f"{output_dir}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Semantic Similarity vs Lexical Diversity by Model
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_success, 
                       x='avg_semantic_similarity', 
                       y='diversity_between_paraphrases',
                       hue='model',
                       style='model',
                       s=100)
        plt.title('Semantic Similarity vs Lexical Diversity by Model')
        plt.savefig(f"{plots_dir}/semantic_vs_lexical.png")
        plt.close()
        
        # 2. Semantic Similarity vs BLEU by Model
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_success, 
                       x='avg_semantic_similarity', 
                       y='avg_bleu',
                       hue='model',
                       style='model',
                       s=100)
        plt.title('Semantic Similarity vs BLEU by Model')
        plt.savefig(f"{plots_dir}/semantic_vs_bleu.png")
        plt.close()
        
        # 3. Semantic Similarity vs ROUGE-L by Model
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_success, 
                       x='avg_semantic_similarity', 
                       y='avg_rougel',
                       hue='model',
                       style='model',
                       s=100)
        plt.title('Semantic Similarity vs ROUGE-1 by Model')
        plt.savefig(f"{plots_dir}/semantic_vs_rouge1.png")
        plt.close()
        
        # 4. Model Performance Comparison (boxplot)
        metrics = ['avg_semantic_similarity', 'diversity_between_paraphrases', 
                  'avg_bleu', 'avg_rougel']
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Model Performance Comparison')
        
        for ax, metric in zip(axes.flat, metrics):
            sns.boxplot(data=df_success, x='model', y=metric, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/model_comparison.png")
        plt.close()
        
        # Generate summary statistics
        summary = df_success.groupby('model')[
            ['avg_semantic_similarity', 'diversity_between_paraphrases', 
             'avg_bleu', 'avg_rougel']].describe()
        summary.to_csv(f"{output_dir}/summary_statistics.csv")
        
        # Add call to analyze best parameters
        self.analyze_best_parameters(df, output_dir)
        return summary

    def analyze_best_parameters(self, df: pd.DataFrame, output_dir: str = "experiment_results"):
        """
        Analyze and identify the best parameter combinations for each model
        
        Args:
            df: DataFrame containing experimental results
            output_dir: Directory to save analysis results
        """
        # Filter successful experiments
        df_success = df[df['success']]
        
        # Define metrics to optimize
        metrics = {
            'avg_semantic_similarity': 'max',     # Higher is better (preserve meaning)
            'diversity_between_paraphrases': 'moderate',  # Moderate is better (balanced rewording)
            'avg_bleu': 'moderate',          # Moderate is better (balanced similarity)
            'avg_rougel': 'moderate'         # Moderate is better (balanced overlap)
        }
        
        # Define target values for 'moderate' metrics
        moderate_targets = {
            'diversity_between_paraphrases': 0.6,  # Aim for 60% unique words
            'avg_bleu': 0.5,          # Aim for 50% BLEU score
            'avg_rougel': 0.5         # Aim for 50% ROUGE score
        }
        
        results = []
        
        for model in df_success['model'].unique():
            model_df = df_success[df_success['model'] == model]
            
            # Find best parameters for each metric
            best_params = {}
            
            for metric, objective in metrics.items():
                if objective == 'max':
                    # For metrics where higher is better
                    best_row = model_df.loc[model_df[metric].idxmax()]
                elif objective == 'moderate':
                    # For metrics where moderate values are better
                    target = moderate_targets[metric]
                    model_df['distance_from_target'] = abs(model_df[metric] - target)
                    best_row = model_df.loc[model_df['distance_from_target'].idxmin()]
                
                # Collect parameters dynamically
                best_params[metric] = {
                    'value': best_row[metric],
                    'target': moderate_targets.get(metric, 'maximize'),
                    'parameters': {
                        'temperature': best_row.get('temperature', None),
                        'repetition_penalty': best_row.get('repetition_penalty', None),
                        'top_p': best_row.get('top_p', None)
                    }
                }
            
            # Find overall best parameters (balanced approach)
            normalized_df = model_df.copy()
            for metric in metrics.keys():
                if metrics[metric] == 'moderate':
                    # For moderate metrics, closer to target is better
                    target = moderate_targets[metric]
                    normalized_df[f'{metric}_score'] = 1 - abs(normalized_df[metric] - target) / target
                else:
                    # For maximize metrics, higher is better
                    normalized_df[f'{metric}_score'] = (normalized_df[metric] - normalized_df[metric].min()) / \
                        (normalized_df[metric].max() - normalized_df[metric].min())
            
            # Calculate composite score
            normalized_df['composite_score'] = normalized_df[[f'{m}_score' for m in metrics.keys()]].mean(axis=1)
            
            # Get best overall parameters
            best_overall = normalized_df.loc[normalized_df['composite_score'].idxmax()]
            
            results.append({
                'model': model,
                'best_overall_parameters': {
                    'temperature': best_overall.get('temperature', None),
                    'repetition_penalty': best_overall.get('repetition_penalty', None),
                    'top_p': best_overall.get('top_p', None)
                },
                'best_overall_scores': {
                    metric: {
                        'value': best_overall[metric],
                        'target': moderate_targets.get(metric, 'maximize')
                    } for metric in metrics.keys()
                },
                'composite_score': best_overall['composite_score'],
                'best_by_metric': best_params
            })
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/best_parameters.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create parameter analysis plots
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics.keys(), 1):
            plt.subplot(2, 2, i)
            sns.scatterplot(data=df_success, 
                           x='temperature', 
                           y=metric, 
                           hue='model',
                           sizes=(50, 200))
            plt.title(f'Temperature vs {metric}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/parameter_analysis.png")
        plt.close()
        
        # Print summary
        print("\nBest Parameters Summary:")
        print("=" * 80)
        for result in results:
            print(f"\nModel: {result['model']}")
            print(f"Best Overall Parameters:")
            for param, value in result['best_overall_parameters'].items():
                if value is None:
                    print(f"  {param}: {value}")
                else:
                    print(f"  {param}: {value:.2f}")
            print(f"Resulting Scores:")
            for metric, value in result['best_overall_scores'].items():
                print(f"  {metric}: {value['value']:.3f}")
            print(f"Target: {value['target']}")
            print(f"Composite Score: {result['composite_score']:.3f}")
        