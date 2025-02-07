# pip install nlpaug

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw

phrases = [
    "Write Python code for addition, subtraction, division, multiplication, and other similar operations, all a part of one class",
    "Given the business logic code below, implement Flask backend. Do not include example of usage or the business logic, do not repeat any code from this prompt. Only write the Flask API code.",
]


for phrase in phrases:
    print("Original:")
    print(phrase)
    # aug = nac.KeyboardAug(aug_char_p=0.1, aug_word_p=1.0, aug_word_max=len(phrase))
    # augmented_texts = aug.augment(phrase, n=1)
    # print("Keyboard aug:")
    # print(augmented_texts)

    # wea = naw.WordEmbsAug(
    #     model_type='word2vec', model_path=model_dir+'GoogleNews-vectors-negative300.bin',
    #     action="substitute")
    # augmented_texts = wea.augment(phrase, n=1)
    # print("Word embedding aug:")
    # print(augmented_texts)
    # aug = naw.SpellingAug(aug_p=0.5)
    # augmented_texts = aug.augment(phrase, n=1)
    # print("Spelling aug:")
    # print(augmented_texts)

    aug = naw.SynonymAug(aug_p=0.5)
    augmented_texts = aug.augment(phrase, n=1)
    print("Spelling aug:")
    print(augmented_texts)
