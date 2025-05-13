from .base import BaseAugmenter

class ParaphraserAugmenter(BaseAugmenter):
    def __init__(self, rate, paraphrases_file):
        super().__init__("Paraphraser", rate)

        if rate < 0.001:
            # special case, make sure we return unaltered text for 0
            # don't compare to 0.0 directly to avoid precision issues
            self.is_no_op = True
        else:
            rate = 1 - rate # convert from aug rate to bleu score
            self.range = self._determine_range(rate)
            self.paraphrases_df = self._load_paraphrases(paraphrases_file)
            self.is_no_op = False
    
    def _load_paraphrases(self, paraphrases_file):
        import pandas as pd
        paraphrases_df = pd.read_csv(paraphrases_file)
        return paraphrases_df
    
    def _determine_range(self, rate):
        ranges = [ # Ranges for the sacre BLEU score  (original, low, moderate, high)
            (1.0-0.001, 1.0), # original, unused
            (0.5, 1.0-0.001), # low variation band
            (0.2, 0.5), # medium variation band
            (0.0, 0.2), # high variation band
        ] 
        for r in ranges:
            if r[0] <= rate <= r[1]:
                return r

    def augment(self, text):
        if self.is_no_op:
            return text

        all_paraphrases = self.paraphrases_df[self.paraphrases_df['original_phrase'] == text].copy()
        # Handle the case when original phrase is not found
        if all_paraphrases.shape[0] == 0:
            return None

        in_range_paraphrases = all_paraphrases[(all_paraphrases['sacre_bleu'] >= self.range[0]) & (all_paraphrases['sacre_bleu'] <= self.range[1])]
        if in_range_paraphrases.empty:
            return None
        selected_paraphrase = in_range_paraphrases.sample(1).iloc[0]['paraphrase']
        return selected_paraphrase
    
    def get_original_prompts(self):
        return self.paraphrases_df['original_phrase'].unique()
    

    def get_bleu_score(self, paraphrase):
        if paraphrase in self.get_original_prompts():
            return 0.0

        record = self.paraphrases_df[self.paraphrases_df['paraphrase'] == paraphrase].unique()
        return record['sacre_bleu']
