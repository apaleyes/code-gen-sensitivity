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
* [sandbox](sandbox/) contains different dabbles, small experiments and API tries. [dataset](sandbox/dataset.yml) provides some data for it. [multi-stage-pipeline](sandbox/multi-stage-pipeline.txt) is an example of what multi-stage code generate pipeline with LLM could look like. It was created with ChatGPT.
* [Main experiment](main_experiment.py) does the augment-generate-measure test. It saves a JSON file with inputs and metrics that [analysis](analysis.ipynb) looks at. LLSs currently supported can be found in [models](models/) folder.
