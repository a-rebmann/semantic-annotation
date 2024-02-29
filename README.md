# Enabling Semantics-aware Process Mining through the Automatic Annotation of Event Logs.
<sub>
written by <a href="mailto:rebmann@informatik.uni-mannheim.de">Adrian Rebmann</a><br />
</sub>

## About
This repository contains the prototype of the approach, evaluation, training data, and gold standards as described in

*Rebmann, A., & van der Aa, H. Enabling Semantics-aware Process Mining through the Automatic Annotation of Event Logs. <br>Published in Information Systems*

<b>If you want to use the approach or parts of it in your project we refer to our python package [here](https://gitlab.uni-mannheim.de/processanalytics/semantic-role-extraction) for more convenient usage.</b>

## Setup and Usage

### Installation instructions
**The project was developed in python 3.9**

1. create a virtual environment for the project 
2. install the dependencies in requirements.txt: using pip <code> pip install -r requirements.txt </code>
3. The approach uses the POS-tagger from spacy => run <code>python -m spacy download en_core_web_lg</code>
4. To install WordNet (used for resource categorization) run <code>nltk.download()</code> from a python console in your virtual environment or add it to the top of <code>main.py</code>

### Input, Output and Model directories
The following default directories are used for input and output, as well as trained models used by the approach:
* DEFAULT_INPUT_DIR = 'input/logs/' (includes a small CSV sample log)
* DEFAULT_OUTPUT_DIR = 'output/logs/' 
* DEFAULT_MODEL_DIR = '.model/main/'
* DEFAULT_RES_DIR = 'resources/'

You'll have to adapt DEFAULT_INPUT_DIR and DEFAULT_OUTPUT_DIR in <code>main.py</code>, if you want to load your input (and write your output) from (to) a different location

Place your input into DEFAULT_INPUT_DIR.

Note that all the serialized models and configuration files in <code>'.model/main/'</code> (includes our fine-tuned BERT) and the content of <code>resources/</code> (contains Glove-Embeddings, resource classifier, and the reference actions for action type categorization) are necessary for the project to run 

### Usage 
1. Configure the parameters in  <code>main.py</code> (if needed)
    * confidence threshold for the attribute classification: <code>confidence_threshold</code>
    * output style of the event log: <code>expanded</code>, default is true. Expanded style creates a attribute per role instance of an event, i.e. if an event has two actions there will be two additional attributes <code>action:name:0, action:name:1</code>. Otherwise there is one attribute per role, e.g. <code>action:name</code>, i.e. if there are multiple instances for a role, this is recorded as a collection.
    * ratio of attribute values tagged as business object to reassign to the attribute classification step: <code>bo_ratio</code>
2. Run the project using <code>python main.py</code>

## References
* [pm4py](https://pm4py.fit.fraunhofer.de)
* [BERT](https://github.com/google-research/bert)
* [spacy](https://spacy.io)
* [GloVe embeddings](https://nlp.stanford.edu/projects/glove/)
* [UPOS tag set](https://universaldependencies.org/docs/u/pos/)

## Evaluation
Our approach was evaluated on a range of real world event logs availbale from the 4TU data repository [4TU-Real-life-event-logs](https://data.4tu.nl/search?q=:keyword:%20%22real%20life%20event%20logs%22) <br>
Specifically, the logs in the table below were used.

| Log | Link |
| ------ | ------ |
| BPI Challenge '12 | https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f |
| BPI Challenge '13 closed incidents | https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11 |
| BPI Challenge '14 Acitivity log | https://doi.org/10.4121/uuid:86977bac-f874-49cf-8337-80f26bf5d2ef | 
| BPI Challenge '15 Municipality 1 | https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17 | 
| BPI Challenge '17 | https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b | 
| BPI Challenge '18 | https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972 | 
| BPI Challenge '19 | https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1 | 
| BPI Challenge '20 Permit log | https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550 | 
| Conformance Checking Challenge '19 | https://doi.org/10.4121/uuid:c923af09-ce93-44c3-ace0-c5508cf103ad | 
| Credit Requirements | https://doi.org/10.4121/uuid:453e8ad1-4df0-4511-a916-93f46a37a1b5 | 
| Hospital Billing | https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741 | 
| Road Traffic Fine Management | https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5 | 
| Sepsis cases | https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460 | 
| WABO Receipt phase | https://doi.org/10.4121/uuid:26aba40d-8b2d-435b-b5af-6d4bfbd7a270 | 

For reproducability we included the gold standard for instance-level labeling, here: <code>input/evaluation/gold/</code><br>.
The gold standard for attribute classification is here: <code>resources/attributes_ground_truth.json</code><br>.
The gold standard for action categorization is here: <code>resources/gt_actions.json</code><br>.
The gold standard for actor categorization is here: <code>resources/resource_ground_truth.json</code><br>.
The data used for fine-tuning the language model is also included, here: <code>model/main/ACTIVITIES.txt</code> (process model activities) and here: <code>model/main/MORE_ACTIVITIES.txt</code> (textual process descriptions)


We conducted a leave-one-out cross validation for the semantic component identification, which is why we fine-tuned the language model in 14 versions 
(each has a size of approx. 500MB, so we cannot publish all of them in the repository. You can train them yourself using the
code here:
[Fine-tuning BERT for semantic labeling](https://gitlab.uni-mannheim.de/processanalytics/fine-tuning-bert-for-semantic-labeling)
and the gold standard of the logs in input/evaluation/gold/
for the comparative evaluation, we included the model
trained on the same data as the HMM baseline in model/same_as_hmm

