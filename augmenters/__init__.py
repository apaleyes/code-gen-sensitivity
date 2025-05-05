def get_augmenter(augmenter_name, rate, **kwargs):
    if augmenter_name.lower() == "keyboard":
        from .keyboard import KeyboardAugmenter

        text_len = kwargs.get("text_len", None)
        return KeyboardAugmenter(rate, text_len)

    if augmenter_name.lower() == "synonym":
        from .synonym import SynonymAugmenter

        text_len = kwargs.get("text_len", None)
        return SynonymAugmenter(rate, text_len)

    if augmenter_name.lower() == "paraphraser":
        from .paraphraser import ParaphraserAugmenter

        paraphrases_file = kwargs["paraphrases_file"]
        return ParaphraserAugmenter(rate, paraphrases_file)

    raise ValueError(f"Unknown augmenter name {augmenter_name}")
