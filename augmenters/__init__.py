def get_augmenter(augmenter_name, rate, paraphrases_file=None):
    if augmenter_name.lower() == "keyboard":
        from .keyboard import KeyboardAugmenter

        return KeyboardAugmenter(rate)

    if augmenter_name.lower() == "synonym":
        from .synonym import SynonymAugmenter

        return SynonymAugmenter(rate)

    if augmenter_name.lower() == "paraphraser":
        from .paraphraser import ParaphraserAugmenter

        return ParaphraserAugmenter(rate, paraphrases_file)

    raise ValueError(f"Unknown augmenter name {augmenter_name}")
