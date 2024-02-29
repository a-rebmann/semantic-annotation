import random

import pandas as pd
import logging

from config import Config
from const import MatchingMode, EVAL_MODE
from evaluate import EVAL_INPUT_DIR, BERT_DIR_ON_MACHINE, EVAL_OUTPUT_DIR
from instancelabeling.bert_tagger.bert_tagger import BertTagger
from main import DEFAULT_RES_DIR

DEFAULT_CONFIG = Config(input_dir=EVAL_INPUT_DIR, model_dir=BERT_DIR_ON_MACHINE, resource_dir=DEFAULT_RES_DIR,
                        output_dir=EVAL_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                        exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.EXACT,
                        mode=EVAL_MODE, res_type=True)

NUMBER_OF_SAMPLES_PER_STRATEGY = 100

def augment_for_resource_classification():
    # Getting the training data
    df = pd.read_csv("/Users/adrianrebmann/Google Drive/data/caise_extensions/" + 'human_system.csv', sep=";")
    df.head()
    col = 'activity'

    artificial_training_data = set()

    bert_tagger = BertTagger(config=DEFAULT_CONFIG)
    unique_activities = df.loc[(df["process_name"] == "General") & (df["task_type"] == "SYS")][col].unique()
    tagged_instances = []
    actions = []
    bos = []
    for act in unique_activities:
        lab = bert_tagger.predict_single_label(act)
        print(lab)
        tagged_instances.append([(tok, tag) for tok, tag in zip(lab[0], lab[1])])
    for item in tagged_instances:
        last = ('X', 'X')
        for tup in item:
            print(tup)
            if last[1] == 'X':
                last = (tup[0], tup[1])
            elif last[1] == tup[1]:
                last = (last[0] + ' ' + tup[0], tup[1])
            else:
                if tup[1] == 'A':
                    actions.append(tup[0])
                if tup[1] == 'BO':
                    bos.append(tup[0])
                last = (tup[0], tup[1])
            if tup[1] == 'A':
                actions.append(tup[0])
            if tup[1] == 'BO':
                bos.append(tup[0])

    # # shuffle existing sentences
    # for i in range(NUMBER_OF_SAMPLES_PER_STRATEGY):
    #     s = []
    #     action = random.choice(actions)
    #     bo = random.choice(bos)
    #     for part in bo.split():
    #         s.append((part, 'BO'))
    #     for part in action.split():
    #         s.append((part, 'A'))
    #
    #     artificial_training_data.add(tuple(s))

    # shuffle existing sentences
    for i in range(NUMBER_OF_SAMPLES_PER_STRATEGY):
        s = []
        action = random.choice(actions)
        bo = random.choice(bos)
        for part in action.split():
            s.append((part, 'A'))
        for part in bo.split():
            s.append((part, 'BO'))

        artificial_training_data.add(tuple(s))

    res_df = pd.DataFrame()
    res_df[col] = pd.Series([" ".join([tup[0] for tup in item]) for item in artificial_training_data])
    res_df["task_type"] = "SYS"
    res_df["process_name"] = "General"
    res_df.to_csv(DEFAULT_CONFIG.resource_dir+"aug_resource_labs.csv", sep=";")


if __name__ == '__main__':
    augment_for_resource_classification()