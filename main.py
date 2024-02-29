import os
import sys
import time

from attributeclassification.actionclassification import ActionClassifier
from attributeclassification.resourceclassifier import ResourceClassifier
from readwrite import writer, loader
from data.word_embeddings import WordEmbeddings
from attributeclassification.attribute_classification import AttributeClassifier
from model.augmented_log import AugmentedLog
import preprocessing.preprocessor as pp
from instancelabeling.bert_tagger.bert_tagger import BertTagger
from config import Config
from const import MatchingMode, MAIN_MODE

# DIRECTORIES
# default input directory
DEFAULT_INPUT_DIR = 'input/logs/'
# default output directory
DEFAULT_OUTPUT_DIR = 'output/logs/'
# default directory where the models are stored
DEFAULT_MODEL_DIR = '.model/main/'
# default directory for resources
DEFAULT_RES_DIR = 'resources/'

DEFAULT_CONFIG = Config(input_dir=DEFAULT_INPUT_DIR, model_dir=DEFAULT_MODEL_DIR, resource_dir=DEFAULT_RES_DIR,
                        output_dir=DEFAULT_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                        exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.SEM_SIM,
                        mode=MAIN_MODE, res_type=True)


def add_resource_and_action_types(aug_log, res_to_type, act_to_type):
    cls_to_res = {}
    for res, type in res_to_type.items():
        if type not in cls_to_res:
            cls_to_res[type] = []
        cls_to_res[type].append(res)
    cls_to_act = {}
    for act, type in act_to_type.items():
        if type not in cls_to_act:
            cls_to_act[type] = []
        cls_to_act[type].append(act)

    aug_log.augmented_df["action:type"] = ""
    aug_log.augmented_df["resource:type"] = ""
    for act_type, acts in cls_to_act.items():
        aug_log.augmented_df.loc[aug_log.augmented_df.isin(acts).any(axis=1), "action:type"] = act_type
    for res_type, reses in cls_to_res.items():
        aug_log.augmented_df.loc[aug_log.augmented_df.isin(reses).any(axis=1), "resource:type"] = res_type
    aug_log.augmented_df.drop('main_resource', inplace=True, axis=1)


def augment_and_transform_log(directory, name, config):
    print(name)
    aug_log = loader.deserialize_event_log(config.resource_dir, name)
    # STEP 1 loading the log and doing semantic component extraction
    if aug_log is False:
        tic = time.perf_counter()
        df, case_id, event_label, time_stamp = loader.load_log(directory, name)
        aug_log = AugmentedLog(name, df, case_id, event_label=event_label, timestamp=time_stamp)
        toc = time.perf_counter()
        print(f"Loaded the current log in {toc - tic:0.4f} seconds")
        tic = time.perf_counter()
        pp.pre_process(config, aug_log)
        toc = time.perf_counter()
        print(f"Preprocessed the current log in {toc - tic:0.4f} seconds")
        writer.serialize_event_log(config.resource_dir, aug_log)
    print("load word embeddings " + config.word_embeddings_file)
    # STEP 1 Extracting Object types and actions
    print(f"Step 1: extracting type and action info")
    tic_1 = time.perf_counter()
    word_embeddings = WordEmbeddings(config=config)
    if aug_log.augmented_df is None:
        print("BERT-based semantic tagging")
        print("obtain .model")
        tic = time.perf_counter()
        bert_tagger = BertTagger(config=config)
        toc = time.perf_counter()
        print(f"Loaded the trained .model in {toc - tic:0.4f} seconds")
        print('semantic tagging text attributes')
        bert_tagger.get_tags_for_df(aug_log)
        for key, val in aug_log.tagged_labels.items():
            print(key, val)
        toc = time.perf_counter()
        print(f"Tagged the whole data set in {tic - toc:0.4f} seconds")
        tic = time.perf_counter()
        print('starting attribute classification')
        attribute_classifier = AttributeClassifier(config=config, word_embeddings=word_embeddings)
        attribute_classifier.run(aug_log=aug_log, bert_tagger=bert_tagger)
        toc = time.perf_counter()
        print(f"Attribute classification finished within {toc - tic:0.4f} seconds")
        print('starting action categorization')
        tic = time.perf_counter()
        action_classifier = ActionClassifier(config, aug_log=aug_log, embeddings=word_embeddings)
        act_to_type = action_classifier.classify_actions()
        toc = time.perf_counter()
        print(f"Categorized actions in {toc-tic:0.4f} seconds")
        aug_log.to_result_log_full(expanded=True, add_refined_label=False)
        print('starting actor categorization')
        tic = time.perf_counter()
        res_class = ResourceClassifier(config, aug_log)
        res_to_type, _, _, _, _, _, _ = res_class.classify_resources()
        toc = time.perf_counter()
        print(f"Categorized actors in {toc - tic:0.4f} seconds")
        add_resource_and_action_types(aug_log, res_to_type, act_to_type)
        writer.serialize_event_log(config.resource_dir, aug_log)
        writer.create_file_for_df(aug_log.augmented_df, DEFAULT_OUTPUT_DIR, aug_log.name)
    toc = time.perf_counter()
    print(f"Finished within {toc - tic_1:0.4f} seconds")


def main():
    list_of_files = {}
    for (dir_path, dir_names, filenames) in os.walk(DEFAULT_CONFIG.input_dir):
        for filename in filenames:
            if filename.endswith('.xes') or (filename.endswith('.csv') and '_info' not in filename):
                list_of_files[filename] = os.sep.join([dir_path])

    print(list_of_files)
    for key, value in list_of_files.items():
        augment_and_transform_log(value, key, DEFAULT_CONFIG)


if __name__ == '__main__':
    main_tic = time.perf_counter()
    main()
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()
