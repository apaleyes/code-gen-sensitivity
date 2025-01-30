from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import spacy
import numpy as np
from typing import List, Dict
import nltk
import warnings


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
        bleu = sentence_bleu([original_tokens], paraphrase_tokens)

        # Calculate ROUGE Scores
        # ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate the quality of text summarization and machine translation algorithms.
        # It measures the overlap between the generated summary and the reference summary based on n-grams.
        rouge_scores = self.rouge_scorer.score(original, paraphrase)

        # Calculate Semantic Similarity using spaCy
        # Semantic similarity measures the degree of similarity in meaning between two texts.
        # In this case, it's calculated using the cosine similarity between the vector representations of the two texts.
        doc1 = self.nlp(original)
        doc2 = self.nlp(paraphrase)
        semantic_similarity = doc1.similarity(doc2)

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
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'semantic_similarity': semantic_similarity,
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
        
        for idx, paraphrase_dict in enumerate(paraphrases, 1):
            paraphrase_text = paraphrase_dict['phrase']
            approach = paraphrase_dict.get('approach', 'unknown')
            model = paraphrase_dict.get('model', 'unknown')
            
            # Get all metrics
            metrics = self.evaluate_single_paraphrase(original, paraphrase_text)
            readability = self.evaluate_readability(paraphrase_text)
            grammar = self.evaluate_grammar(paraphrase_text)
            
            # Combine all metrics
            result = {
                'paraphrase_id': idx,
                'original': original,
                'text': paraphrase_text,
                'approach': approach,
                'model': model,
                **metrics,
                'readability': readability,
                'grammar': grammar
            }
            
            results.append(result)
        
        # Calculate aggregate metrics
        avg_metrics = {
            'avg_bleu': np.mean([r['bleu'] for r in results]),
            'avg_rouge1': np.mean([r['rouge1'] for r in results]),
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
            print(f"ROUGE-1 F1: {result['rouge1']:.3f}")
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
        print(f"Average ROUGE-1: {agg['avg_rouge1']:.3f}")
        print(f"Average Semantic Similarity: {agg['avg_semantic_similarity']:.3f}")
        print(f"Diversity Between Paraphrases: {agg['diversity_between_paraphrases']:.3f}")