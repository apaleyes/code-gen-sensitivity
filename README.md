# Code generation sensitivity

This repository accompanies paper TODO: title here

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
* [Main experiment](main_experiment.py) does the augment-generate-measure test. It saves a JSON file with inputs and metrics that [analysis](analysis.ipynb) looks at. LLMs currently supported can be found in [models](models/) folder. Note that this is only an example of a synthetic evaluation pipeline that hopefully illustrated the idea.
* [experimental_setup](experimental_setup/) contains implementation of the main experiment, expanded over multiple models, datasets and augmentation methods. Any folder starting with "augmented_datasets" prefix contains results, while other files implement steps of the pipeline: augment datasets -> generate LLM responses -> compute metrics -> create charts.
  * Pipeline
    * augment_datasets: take in the datasets jsons, and produce new ones with additional versions of the tasks augmented to 10 different level.
    * get_llm_responses: take each question in the augmented dataset and ask it of multiple llms, noting responses.
    * get_experiment_scores: using the responses, calculate key metrics like TSED.
    * get_experiment_charts: charting logic summarising the experiment scores.
  * Folders containing intermediate results corresponding to the above stages
  * Helper functionality like LLM access, original datasets folder, additional plotting logic

* [personas_experiments](personas_experiments/) contains all code necessary to replicate persona related experiments.
* [Dataset](Sensitivity%20of%20LLMs%20tasks%20dataset.md) is the dataset we compiled specifically for this study.
* [test](tests/) unit tests for some key functions.
* [sandbox](sandbox/) contains different dabbles, small experiments and API tries. [dataset](sandbox/dataset.yml) provides some data for it. [multi-stage-pipeline](sandbox/multi-stage-pipeline.txt) is an example of what multi-stage code generate pipeline with LLM could look like. It was created with ChatGPT.
