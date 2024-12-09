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
* Any *_test file is just a snippet for trying some package/API. For example, [augmentation_test](augmentation_test.py) showcases NLPaug library we use to augment prompts
* [dataset](dataset.yml) provides some testing data.
* [multi-stage-pipeline](multi-stage-pipeline.txt) is and example of what multi-stage code generate pipeline with LLM could look like. It was created with ChatGPT UI.
* [Main experiment](main_experiment.py) implements an experiment that measures an LLM's sensitivity. It saves a csv file that [analysis](analysis.ipynb) looks at. `models` folder implements intefaces for calling LLM model APIs.
