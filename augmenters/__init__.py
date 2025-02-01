def get_augmenter(augmenter_name, rate):
    if augmenter_name.lower() == "keyboard":
        from .keyboard import KeyboardAugmenter
        return KeyboardAugmenter(rate)

    if augmenter_name.lower() == "synonym":
        from .synonym import SynonymAugmenter
        return SynonymAugmenter(rate)

    raise ValueError(f"Unknown augmenter name {augmenter_name}")
