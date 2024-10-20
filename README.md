# Code generation sensitivity

To install dependencies:
1. (Recommended) Create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Clone https://github.com/Etamin/TSED, you should end up with TSED folder. For example:
```
git clone https://github.com/Etamin/TSED.git
```

Guided tour:
* [augmentation_test](augmentation_test.py) and [code_distance_test](code_distance_test.py) showcase key dependencies: NLPaug library we use to augment prompts, and TSED method we use to calculate code distance. [dataset](dataset.yml) provides some data for the latter test.
* [multi-stage-pipeline](multi-stage-pipeline.txt) is and example of what multi-stage code generate pipeline with LLM could look like. It was created with ChatGPT.
* [Gemini test](gemini_augmentation_test.py) does the augment-generate-measure test using Google's Gemini. It saves a csv file that [analysis](analysis.ipynb) looks at.
