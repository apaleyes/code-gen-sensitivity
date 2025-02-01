import nlpaug.augmenter.word as naw

from .base import BaseAugmenter

class SynonymAugmenter(BaseAugmenter):
    def __init__(self, rate):
        super().__init__("Synonym", rate)

        self.augmenter = naw.SynonymAug(aug_p=rate,
                                        aug_min=0, aug_max=None)

    def augment(self, text):
        return self.augmenter.augment(text, n=1)[0]
