# To run this demo, create a python environment (venv or conda)
#
# clone https://github.com/Etamin/TSED , this creates a folder called TSED
#
# then install dependecies (vesion has to be pinned because of https://github.com/grantjenks/py-tree-sitter-languages/issues/64)
# > pip install apted tree_sitter_languages tree-sitter==0.21.3 pyyaml
#
# all is ready, run
# > python test.py

import warnings

warnings.filterwarnings("ignore")

import yaml

from TSED import TSED
import torchmetrics

with open("sandbox/dataset.yml", "r") as dataset_file:
    dataset = yaml.safe_load(dataset_file)

print(dir(torchmetrics.text))

bert_score_metric = torchmetrics.text.BERTScore(model_name_or_path="roberta-large")

original_example = next(x for x in dataset if x["is_original"])
for example in dataset:
    if example["is_original"]:
        continue 

    tree_similarity_score = TSED.Calaulte(
        "python", original_example["answer"], example["answer"], 1.0, 0.8, 1.0
    )
    bert_score = bert_score_metric([example["answer"]],[original_example["answer"]])

    print(
        f"Tree score: {tree_similarity_score:.2};\t\tChange description: {example['how_changed']}",
        f"BERT score: {bert_score:.2};\t\tChange description: {example['how_changed']}"
    )
