import csv
import time
import sys
import os
from copy import deepcopy

import pandas as pd

import instancelabeling.labelparser.labelparser as hmm
import instancelabeling.bert_tagger.bert_preprocessor as bpp
from attributeclassification.actionclassification import ActionClassifier
from attributeclassification.attribute_classification import AttributeClassifier
from attributeclassification.resourceclassifier import ResourceClassifier
from config import Config
from model.augmented_log import AugmentedLog
from data.word_embeddings import WordEmbeddings
from preprocessing.nlp_utils import STOP_WORDS
from const import MatchingMode, type_mapping, EVAL_MODE, AttributeType, ACTION_IDX_TO_LABEL, ConceptType, SCHEMA_ORG
from evaluation.baseline_eval_reult import BaselineEvalResult
from evaluation.bert_eval_result import BertEvalResult
from evaluation.chunkeval import ChunkEval
from evaluation.metrics import classification_report, conf_matrix
from preprocessing.preprocessor import preprocess_label
from readwrite import loader, writer
from main import DEFAULT_RES_DIR, pp, BertTagger

log_to_label = {
    "BPIC15_1.xes_gs.csv": ["activityNameEN"],
    "Detail_Incident_Activity.csv_gs.csv": ["IncidentActivity_Type"],
    "CCC19 - Log XES.xes_gs": ["ACTIVITY", "concept:name"],
    "BPI_Challenge_2018.xes": ["concept:name", "activity"]
}

log_to_label_raw = {
    "BPIC15_1.xes": ["activityNameEN"],
    "Detail_Incident_Activity.csv": ["IncidentActivity_Type"],
    "CCC19 - Log XES.xes": ["ACTIVITY", "concept:name"],
    "BPI_Challenge_2018.xes": ["concept:name", "activity"]
}

tagged_logs = [
    "BPI_Challenge_2012.xes_gs.csv",
    "BPIC15_1.xes_gs.csv",
    "BPI_Challenge_2018.xes_gs.csv",
    "BPI_Challenge_2019.xes_gs.csv",
    "BPI_Challenge_2017.xes_gs.csv",
    "PermitLog.xes_gs.csv",
    "CreditRequirement.xes_gs.csv",
    "CCC19 - Log XES.xes_gs.csv",
    "BPI_Challenge_2013_closed_problems.xes_gs.csv",
    "Detail_Incident_Activity.csv_gs.csv",
    "Receipt phase of an environmental permit application process ( WABO ) CoSeLoG project.xes_gs.csv",
    "Road_Traffic_Fine_Management_Process.xes_gs.csv",
    "Sepsis Cases - Event Log.xes_gs.csv",
    "Hospital Billing - Event Log.xes_gs.csv",
    "p2p.jsonocel_gs.csv",
    "o2c.jsonocel_gs.csv",
    "runningexample.jsonocel_gs.csv"
]

tagged_logs_raw = [
    "BPI_Challenge_2012.xes",
    "BPIC15_1.xes",
    "BPI_Challenge_2018.xes",
    "BPI_Challenge_2019.xes",
    "BPI_Challenge_2017.xes",
    "PermitLog.xes",
    "CreditRequirement.xes",
    "CCC19 - Log XES.xes",
    "BPI_Challenge_2013_closed_problems.xes",
    "Detail_Incident_Activity.csv",
    "Receipt phase of an environmental permit application process ( WABO ) CoSeLoG project.xes",
    "Road_Traffic_Fine_Management_Process.xes",
    "Sepsis Cases - Event Log.xes",
    "Hospital Billing - Event Log.xes",
    "p2p.jsonocel",
    "o2c.jsonocel",
    "runningexample.jsonocel"
]

EVAL_INPUT_DIR = 'input/evaluation/'
HMM_DIR = "instancelevellabeling/labelparser/trainedparser.p"
HMM_BASE_DIR = "instancelevellabeling/labelparser/"

################################################################################################
# the raw logs used for evaluation are listed in tagged_logs_raw
# and are available here:
# https://data.4tu.nl/search?q=:keyword:%20%22real%20life%20event%20logs%22
# place them in the input/evaluation/raw/ folder
# We conducted a leave-one-out cross validation which is why we
# fine-tuned the language model in 14 versions
# (each has a size of approx. 500MB, so we cannot publish all of
# them in the repository. You can train them yourself using the
# code here:
# https://gitlab.uni-mannheim.de/process-analytics-public/fine-tuning-bert-for-semantic-labeling
# and the gold standard of the logs in input/evaluation/gold/
# for the comparative evaluation, we included the model
# trained on the same data as the HMM baseline in .model/same_as_hmm
#################################################################################################
BERT_DIR_ON_MACHINE = '/Users/adrianrebmann/Develop/semantic_event_parsing/fine-tuning-bert-for-semantic-labeling/model/'

EVAL_OUTPUT_DIR = 'output/evaluation/'

ON_SERVER = False
CHUNK_WISE = True

DEFAULT_CONFIG = Config(input_dir=EVAL_INPUT_DIR, model_dir=BERT_DIR_ON_MACHINE, resource_dir=DEFAULT_RES_DIR,
                        output_dir=EVAL_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                        exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.EXACT,
                        mode=EVAL_MODE, res_type=True)

BASELINE_CONFIG = Config(input_dir=EVAL_INPUT_DIR, model_dir=".model/" + "same_as_hmm/",
                         resource_dir=DEFAULT_RES_DIR,
                         output_dir=EVAL_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                         exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.EXACT,
                         mode=EVAL_MODE, res_type=False)

# avoid huggingface warnings due to fork after parallelism was used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_file_names(gold=True):
    list_of_files = {}
    actual_path = EVAL_INPUT_DIR
    if gold:
        actual_path = actual_path + 'gold/'
    else:
        actual_path = actual_path + 'raw/'
    for (dir_path, dir_names, filenames) in os.walk(actual_path):
        for filename in filenames:
            if filename.endswith('.xes') or (
                    filename.endswith('.csv') and '_info' not in filename) or filename.endswith('.jsonocel'):
                list_of_files[filename] = os.sep.join([dir_path])
    return list_of_files


def eval_overall_bert_performance(eval_configs):
    allowed_tags = ['BO', 'A', 'ACTOR', 'REC', 'ASTATE', 'BOSTATE', 'X']
    results = {}
    prec_weighted_by_support = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0,
                                'X': 0}
    rec_weighted_by_support = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0,
                               'X': 0}
    weighted_by_support = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0, 'X': 0}
    overall_support = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0, 'X': 0}
    print(tagged_logs)
    for key, value in get_file_names().items():
        print(key)
        run = tagged_logs.index(key)
        print('this is run number ' + str(run) + ' evaluating with log ' + key)
        for config in eval_configs:
            current_config = deepcopy(config)
            current_config.model_dir = current_config.model_dir + str(run) + '/'
            current_config.exclude_data_origin.append(key)
            tagged_set = pd.read_csv(current_config.input_dir + 'gold/' + key, sep=';', keep_default_na=False)
            bert_tagger = BertTagger(config=current_config)
            consider_col = []
            unique_tagged_labels = bpp.data_to_object(tagged_set, True, consider_col)
            ground_truth = {}
            if CHUNK_WISE:
                for do in unique_tagged_labels:
                    ground_truth[do.label] = [tag for tok, tag in zip(do.split, do.tags) if tok not in STOP_WORDS]
            else:
                for do in unique_tagged_labels:
                    ground_truth[do.label] = [tag for tag in do.tags]
            raw_labels = ground_truth.keys()
            bert_parsed = bert_tagger.get_tags_for_list(list(raw_labels))
            if CHUNK_WISE:
                for do in bert_parsed.keys():
                    bert_parsed[do] = [tag for tok, tag in zip(do.split(), bert_parsed[do]) if tok not in STOP_WORDS]
            if len(ground_truth) > 0:
                if CHUNK_WISE:
                    results[key] = ChunkEval(ground_truth, bert_parsed)
                else:
                    results[key] = BertEvalResult(ground_truth, bert_parsed)
                    results[key].print_examples(num_examples=40)
                for tag in allowed_tags:
                    if tag in results[key].available_tags:
                        overall_support[tag] += results[key].bert_per_class_metrics[tag]['support']
                        weighted_by_support[tag] += results[key].bert_per_class_metrics[tag]['support'] * \
                                                    results[key].bert_per_class_metrics[tag]['f1-score']
                        prec_weighted_by_support[tag] += results[key].bert_per_class_metrics[tag]['support'] * \
                                                         results[key].bert_per_class_metrics[tag]['precision']
                        rec_weighted_by_support[tag] += results[key].bert_per_class_metrics[tag]['support'] * \
                                                        results[key].bert_per_class_metrics[tag]['recall']
            else:
                print('There are no labels for ' + key)
    print(overall_support)
    for tag in allowed_tags:
        print("Overall Precision")
        print(tag, prec_weighted_by_support[tag] / overall_support[tag])
        print("Overall Recall")
        print(tag, rec_weighted_by_support[tag] / overall_support[tag])
        print("Overall F1")
        print(tag, weighted_by_support[tag] / overall_support[tag])
        print("Overall Support")
        print(tag, overall_support[tag])
    results_file = config.output_dir + "bert_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if CHUNK_WISE:
            header = ['model', 'log'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags] + \
                     ['weighted support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        else:
            header = ['model', 'log', 'entity-level-f1', 'entity-level-prec', 'entity-level-rec', 'conf'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['precision ' + tag for tag in allowed_tags] + \
                     ['recall ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags] + \
                     ['weighted support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        for key in results.keys():
            if CHUNK_WISE:
                writer.writerow(['bert', key] +
                                [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] *
                                 results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags])
            else:
                writer.writerow(['bert', key, results[key].bert_entity_level_f1, results[key].bert_entity_level_prec,
                                 results[key].bert_entity_level_rec, results[key].conf_matrix] +
                                [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] *
                                 results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_tags else 0 for tag in allowed_tags])


def eval_against_baseline(eval_configs):
    allowed_tags = ['BO', 'A', 'X']
    prec_weighted_by_support_bert = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0,
                                     'X': 0}
    rec_weighted_by_support_bert = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0,
                                    'X': 0}
    weighted_by_support_bert = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0, 'X': 0}
    prec_weighted_by_support_hmm = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0,
                                    'X': 0}
    rec_weighted_by_support_hmm = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0,
                                   'X': 0}
    weighted_by_support_hmm = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0, 'X': 0}
    overall_support = {"entity": 0, 'BO': 0, 'A': 0, 'ACTOR': 0, 'REC': 0, 'ASTATE': 0, 'BOSTATE': 0, 'X': 0}
    # hmm.train_parser(HMM_BASE_DIR,"trainedparser.p")
    overall_bert_parsed = {}
    overall_hmm_parsed = {}
    overall_ground_truth = {}
    results = {}
    for key, value in get_file_names().items():
        run = tagged_logs.index(key)
        print('this is the run evaluating with log ' + key)
        for config in eval_configs:
            current_config = deepcopy(config)
            current_config.model_dir = current_config.model_dir
            current_config.exclude_data_origin.append(key)
            tagged_set = pd.read_csv(current_config.input_dir + 'gold/' + key, sep=';', keep_default_na=False)
            bert_tagger = BertTagger(config=current_config)
            hmm_tagger = hmm.load_trained_parser(HMM_DIR)
            consider_col = []
            if key in log_to_label_raw.keys():
                consider_col.extend(log_to_label[key])
            else:
                consider_col.append('concept:name')
                if key in log_to_label.keys():
                    consider_col.extend(log_to_label[key])
            unique_tagged_labels = bpp.data_to_object(tagged_set, True, consider_col)
            ground_truth = {}
            if CHUNK_WISE:
                for do in unique_tagged_labels:
                    ground_truth[do.label] = [tag if tag in allowed_tags else 'A' if (
                            tag == 'ASTATE' or tag == 'BOSTATE' or tag == 'STATE') else 'X' for tok, tag in
                                              zip(do.split, do.tags) if tok not in STOP_WORDS]
            else:
                for do in unique_tagged_labels:
                    ground_truth[do.label] = [tag if tag in allowed_tags else 'A' if (
                            tag == 'ASTATE' or tag == 'BOSTATE' or tag == 'STATE') else 'X' for tag in do.tags]
            raw_labels = ground_truth.keys()
            hmm_parsed = hmm.get_tags_for_list(hmm_tagger, list(raw_labels))
            bert_parsed = bert_tagger.get_tags_for_list(list(raw_labels))
            if CHUNK_WISE:
                for do in bert_parsed.keys():
                    bert_parsed[do] = [tag for tok, tag in zip(do.split(), bert_parsed[do]) if tok not in STOP_WORDS]
                for do in hmm_parsed.keys():
                    hmm_parsed[do] = [tag for tok, tag in zip(do.split(), hmm_parsed[do]) if tok not in STOP_WORDS]
            overall_ground_truth.update(ground_truth)
            overall_hmm_parsed.update(hmm_parsed)
            for label in bert_parsed.keys():
                bert_parsed[label] = [
                    tag if tag in allowed_tags else 'A' if (
                                tag == 'ASTATE' or tag == 'BOSTATE' or tag == 'STATE') else 'X'
                    for tag in
                    bert_parsed[label]]
            overall_bert_parsed.update(bert_parsed)
            if len(ground_truth) > 0:
                if CHUNK_WISE:
                    results['BERT' + key] = ChunkEval(ground_truth, bert_parsed)
                    results['HMM' + key] = ChunkEval(ground_truth, hmm_parsed)
                    for tag in allowed_tags:
                        if tag in results['BERT' + key].available_tags:
                            overall_support[tag] += results['BERT' + key].bert_per_class_metrics[tag]['support']
                            weighted_by_support_hmm[tag] += results["HMM" + key].bert_per_class_metrics[tag][
                                                                'support'] * \
                                                            results["HMM" + key].bert_per_class_metrics[tag]['f1-score']
                            weighted_by_support_bert[tag] += results["BERT" + key].bert_per_class_metrics[tag][
                                                                 'support'] * \
                                                             results["BERT" + key].bert_per_class_metrics[tag][
                                                                 'f1-score']
                            prec_weighted_by_support_hmm[tag] += results["HMM" + key].bert_per_class_metrics[tag][
                                                                     'support'] * \
                                                                 results["HMM" + key].bert_per_class_metrics[tag][
                                                                     'precision']
                            prec_weighted_by_support_bert[tag] += results["BERT" + key].bert_per_class_metrics[tag][
                                                                      'support'] * \
                                                                  results["BERT" + key].bert_per_class_metrics[tag][
                                                                      'precision']
                            rec_weighted_by_support_hmm[tag] += results["HMM" + key].bert_per_class_metrics[tag][
                                                                    'support'] * \
                                                                results["HMM" + key].bert_per_class_metrics[tag][
                                                                    'recall']
                            rec_weighted_by_support_bert[tag] += results["BERT" + key].bert_per_class_metrics[tag][
                                                                     'support'] * \
                                                                 results["BERT" + key].bert_per_class_metrics[tag][
                                                                     'recall']
                else:
                    results[key] = BaselineEvalResult(ground_truth, hmm_parsed, bert_parsed, allowed_tags)
                    results[key].print_summary(key)
                # results[key].print_examples(num_examples=20)
            else:
                print('There are no labels for ' + key)
    print('overall gt ' + str(len(overall_ground_truth.keys())))
    if CHUNK_WISE:
        print(overall_support)
        for tag in allowed_tags:
            if tag == 'X':
                continue
            print("Overall Count")
            print('HMM:', tag, overall_support[tag])
            print('BERT:', tag, overall_support[tag])
            print("Overall Precision")
            print('HMM:', tag, prec_weighted_by_support_hmm[tag] / overall_support[tag])
            print('BERT:', tag, prec_weighted_by_support_bert[tag] / overall_support[tag])
            print("Overall Recall")
            print('HMM:', tag, rec_weighted_by_support_hmm[tag] / overall_support[tag])
            print('BERT:', tag, rec_weighted_by_support_bert[tag] / overall_support[tag])
            print("Overall F1")
            print('HMM:', tag, weighted_by_support_hmm[tag] / overall_support[tag])
            print('BERT:', tag, weighted_by_support_bert[tag] / overall_support[tag])
    else:
        overall_result = BaselineEvalResult(overall_ground_truth, overall_hmm_parsed, overall_bert_parsed, allowed_tags)
        overall_result.print_summary(key)

    results_file = config.output_dir + "baseline_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if CHUNK_WISE:
            header = ['model', 'log'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['precision ' + tag for tag in allowed_tags] + \
                     ['recall ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        else:
            header = ['model', 'log', 'entity-level-f1', 'entity-level-prec', 'entity-level-rec', 'conf'] + \
                     ['f1-score ' + tag for tag in allowed_tags] + \
                     ['precision ' + tag for tag in allowed_tags] + \
                     ['recall ' + tag for tag in allowed_tags] + \
                     ['support ' + tag for tag in allowed_tags]
            writer.writerow(header)
        if CHUNK_WISE:
            writer.writerow(['hmm', 'overall'] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'f1-score'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'precision'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'recall'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['HMM'].bert_per_class_metrics[tag][
                                 'support'] if tag in overall_result['HMM'].available_tags else 0 for tag in
                             allowed_tags])

            writer.writerow(['bert', 'overall'] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'f1-score'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'precision'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'recall'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result['BERT'].bert_per_class_metrics[tag][
                                 'support'] if tag in overall_result['BERT'].available_tags else 0 for tag in
                             allowed_tags])
        else:
            writer.writerow(['hmm', 'overall', overall_result.hmm_entity_level_f1, overall_result.hmm_entity_level_prec,
                             overall_result.hmm_entity_level_rec, overall_result.hmm_conf_matrix] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'f1-score'] if tag in overall_result.available_hmm_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'precision'] if tag in overall_result.available_hmm_tags else 0 for tag in
                             allowed_tags] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'recall'] if tag in overall_result.available_hmm_tags else 0 for tag in allowed_tags] + \
                            [overall_result.hmm_per_class_metrics[tag][
                                 'support'] if tag in overall_result.available_hmm_tags else 0 for tag in allowed_tags])
            writer.writerow(
                ['bert', 'overall', overall_result.bert_entity_level_f1, overall_result.bert_entity_level_prec,
                 overall_result.bert_entity_level_rec, overall_result.bert_conf_matrix] + \
                [overall_result.bert_per_class_metrics[tag][
                     'f1-score'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags] + \
                [overall_result.bert_per_class_metrics[tag][
                     'precision'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags] + \
                [overall_result.bert_per_class_metrics[tag][
                     'recall'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags] + \
                [overall_result.bert_per_class_metrics[tag][
                     'support'] if tag in overall_result.available_bert_tags else 0 for tag in allowed_tags])
        for key in results.keys():
            if CHUNK_WISE:
                if 'HMM' in key:
                    writer.writerow(['hmm', key] +
                                    [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags])
                else:
                    writer.writerow(['bert', key] +
                                    [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags] +
                                    [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                        key].available_tags else 0 for tag in allowed_tags])
            else:
                writer.writerow(['hmm', key, results[key].hmm_entity_level_f1, results[key].hmm_entity_level_prec,
                                 results[key].hmm_entity_level_rec, results[key].hmm_conf_matrix] +
                                [results[key].hmm_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags] +
                                [results[key].hmm_per_class_metrics[tag]['precision'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags] +
                                [results[key].hmm_per_class_metrics[tag]['recall'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags] +
                                [results[key].hmm_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_hmm_tags else 0 for tag in allowed_tags])
                writer.writerow(['bert', key, results[key].bert_entity_level_f1, results[key].bert_entity_level_prec,
                                 results[key].bert_entity_level_rec, results[key].bert_conf_matrix] +
                                [results[key].bert_per_class_metrics[tag]['f1-score'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['precision'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['recall'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags] +
                                [results[key].bert_per_class_metrics[tag]['support'] if tag in results[
                                    key].available_bert_tags else 0 for tag in allowed_tags])


def write_attribute_results(config, results, ground_truths, att_annots, with_samples=False):
    results_file = config.output_dir + "attribute_classification_results_" + time.strftime("%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        header = ['log', 'attribute', 'type', 'true', 'pred', 'approach', 'score']
        writer.writerow(header)
        print(results.keys())
        for key in results.keys():
            trues = []
            preds = []
            curr_res = results[key]
            for att in ground_truths[key].keys():
                if att in curr_res[1].keys():
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[1][att][0],
                                     'semantic similarity', curr_res[1][att][1]])
                if att in curr_res[2].keys():
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[2][att],
                                     'bert masking', 'nan'])
                if att in curr_res[3].keys():
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[3][att],
                                     'bert combined', 'nan'])
                if att in curr_res[4].keys():
                    trues.append(ground_truths[key][att])
                    preds.append(curr_res[4][att])
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[4][att],
                                     'final', 'nan'])
                if att in curr_res[5].keys() and with_samples:
                    writer.writerow([key, att, att_annots[key][att] if att in att_annots[
                        key].keys() else AttributeType.UNKNOWN.value, ground_truths[key][att], curr_res[5][att],
                                     'samples', 'nan'])


def do_basic_extraction(current_config, path_to_log, log_name, att_cls=True):
    aug_log = loader.deserialize_event_log(current_config.resource_dir, log_name)
    if aug_log is False:
        df, case_id, event_label, time_stamp = loader.load_log(path_to_log, log_name)
        aug_log = AugmentedLog(log_name, df, case_id, event_label=event_label, timestamp=time_stamp)
        pp.pre_process(current_config, aug_log)
        #writer.serialize_event_log(current_config.resource_dir, aug_log)
    print("BERT-based semantic tagging")
    print("obtain model")
    tic = time.perf_counter()
    bert_tagger = BertTagger(config=current_config)
    toc = time.perf_counter()
    print(f"Loaded the trained model in {toc - tic:0.4f} seconds")
    print('semantic tagging text attributes')
    bert_tagger.get_tags_for_df(aug_log)
    #for key, val in aug_log.tagged_labels.items():
    #    print(key, val)
    tic = time.perf_counter()
    print(f"Tagged the whole data set in {tic - toc:0.4f} seconds")
    print('load GloVe')
    word_embeddings = WordEmbeddings(config=current_config)
    if att_cls:
        print('starting attribute classification')
        tic = time.perf_counter()
        att_class = AttributeClassifier(config=current_config, word_embeddings=word_embeddings)
        pred_labels = att_class.run(aug_log=aug_log, bert_tagger=bert_tagger)
        toc = time.perf_counter()
        print(f"Attribute classification finished within {toc - tic:0.4f} seconds")
        return aug_log, att_class, pred_labels
    else:
        return aug_log, word_embeddings



def evaluate_resource_classifier(eval_configs):
    res_labels = ["HUM", "SYS", "X"]
    overall_pred = []
    overall_true = []

    overall_pred_ne = []
    overall_true_ne = []
    overall_pred_wn = []
    overall_true_wn = []
    overall_pred_bert = []
    overall_true_bert = []
    overall_pred_other = []
    overall_true_other = []
    ground_truth_res = loader.get_resource_ground_truth(DEFAULT_RES_DIR, file="resources_ground_truth.json")
    for key, value in get_file_names(gold=False).items():
        print(key)

        #if "Permit" not in key:
        #    continue
        run = tagged_logs_raw.index(key)
        ground_truth_att = ground_truth_res[key]
        #print("GT:", ground_truth_att)
        true_labels = {}
        for k, v in ground_truth_att.items():
            if k in res_labels:
                for v_x in v:
                    true_labels[v_x] = k
        for config in eval_configs:
            current_config = deepcopy(config)
            current_config.model_dir = current_config.model_dir + str(run) + '/'
            current_config.exclude_data_origin.append(key)
            aug_log = loader.deserialize_event_log(current_config.resource_dir, key)
            if not aug_log or aug_log.augmented_df is None:
                aug_log, att_class, pred_labels = do_basic_extraction(current_config, value, key)
                aug_log.to_result_log_full(expanded=True, add_refined_label=False)
                writer.serialize_event_log(config.resource_dir, aug_log)
            # for key, val in aug_log.tagged_labels.items():
            #     print(key, val)

            res_class = ResourceClassifier(config, aug_log)
            pred_labels_instance, resource_labels_text, resource_labels_misc, resource_labels_ne, resource_labels_wn, resource_labels_bert, resource_labels_other = res_class.classify_resources()
            #print(len(pred_labels_instance), len(resource_labels_text), len(resource_labels_misc))
            pred = []
            true = []
            for att, p in pred_labels_instance.items():
                if att in true_labels.keys():
                    pred.append(p)
                    true.append(true_labels[att])
                else:
                    pass
                    #print("pred", att, p)
            if len(pred) != 0:
                per_log_report = classification_report(true, pred, output_dict=True)
                #print("All")
                #print(per_log_report)

            pred_ne = []
            true_ne = []
            for att, p in resource_labels_ne.items():
                if att in true_labels.keys():
                    pred_ne.append(p)
                    true_ne.append(true_labels[att])
                else:
                    pass
                    #print("pred", att, p)
            if len(pred_ne) != 0:
                per_log_report = classification_report(true_ne, pred_ne, output_dict=True)
                #print("Attribute")
                #print(per_log_report)

            pred_wn = []
            true_wn = []
            for att, p in resource_labels_wn.items():
                if att in true_labels.keys(): #or att in [preprocess_label(k) for k in true_labels.keys()]:
                    pred_wn.append(p)
                    true_wn.append(true_labels[att])
                else:
                    pass
                    #print("pred", att, p)
            if len(pred_wn) != 0:
                per_log_report = classification_report(true_wn, pred_wn, output_dict=True)
                #print("Attribute")
                #print(per_log_report)

            pred_bert = []
            true_bert = []
            for att, p in resource_labels_bert.items():
                if att in true_labels.keys():
                    pred_bert.append(p)
                    true_bert.append(true_labels[att])
                else:
                    pass
                    #print("pred", att, p)
            if len(pred_bert) != 0:
                per_log_report = classification_report(true_bert, pred_bert, output_dict=True)
                #print("Attribute")
                #print(per_log_report)

            pred_other = []
            true_other = []
            for att, p in resource_labels_other.items():
                if att in true_labels.keys():
                    pred_other.append(p)
                    true_other.append(true_labels[att])
                else:
                    pass
                   # print("pred", att, p)
            if len(pred_other) != 0:
                per_log_report = classification_report(true_other, pred_other, output_dict=True)
                #print("Attribute")
                #print(per_log_report)


            overall_pred.extend(pred)
            overall_true.extend(true)

            overall_pred_ne.extend(pred_ne)
            overall_true_ne.extend(true_ne)

            overall_pred_wn.extend(pred_wn)
            overall_true_wn.extend(true_wn)

            overall_pred_bert.extend(pred_bert)
            overall_true_bert.extend(true_bert)

            overall_pred_other.extend(pred_other)
            overall_true_other.extend(true_other)

    if len(overall_true) == 0 or len(overall_pred) == 0:
        print("No results available!")
        return
    print(classification_report(overall_true, overall_pred))

    try:
        print("NE" + ("-"*30))
        print(classification_report(overall_true_ne, overall_pred_ne))
    except ValueError:
        print("No NE report available")
    try:
        print("WN" + ("-" * 30))
        print(classification_report(overall_true_wn, overall_pred_wn))
    except ValueError:
        print("No WN report available")
    try:
        print("BERT" + ("-" * 30))
        print(classification_report(overall_true_bert, overall_pred_bert))
    except ValueError:
        print("No BERT report available")
    try:
        print("TIME" + ("-" * 30))
        print(classification_report(overall_true_other, overall_pred_other))
    except ValueError:
        print("No TIME report available")

CAiSE_VERSION = False # Must be set also in attribute_classifcation and subclassifiers.att_label_classifier!

def eval_attribute_classification(eval_configs):
    atts_handles_per_step = {"step_1": 0, "step_2": 0, "step_3_cons":0, "step_3_other": 0}
    atts_per_log_handled_per_step = {}
    atts_per_log_count = {}
    atts = []
    overall_pred = []
    overall_true = []
    macro_dict = {label: {} for label in type_mapping.values()}
    macro_dict["X"] = {}
    macro_dict["macro avg"] = {}
    macro_dict["weighted avg"] = {}
    true_count_all = {}

    atts_per_label = {label: set() for label in list(type_mapping.values())+["X"]}
    atts_per_label_per_log = {}
    results_file = DEFAULT_CONFIG.output_dir + "attribute_classification_results_" + time.strftime(
        "%Y%m%d%H%M%S") + ".csv"
    with open(results_file, 'w', newline='') as csvfile:
        write = csv.writer(csvfile, delimiter=';')
        header = ['log', 'attribute', 'true', 'pred', 'step']
        write.writerow(header)
        for key, value in get_file_names(gold=False).items():
            atts_per_log_handled_per_step[key] = {"step_1": [], "step_2": [], "step_3_cons": [], "step_3_other": []}
            atts_per_log_count[key] = 0
            atts_per_log = []
            atts_per_label_per_log[key] = {label: set() for label in list(type_mapping.values()) + ["X"]}
            print(key)
            #if "BPI_Challenge_2018" not in key:
            #    continue
            run = tagged_logs_raw.index(key)
            ground_truth = loader.get_annotated_attributes(DEFAULT_RES_DIR, file="attributes_ground_truth.json")[key]
            true_labels = {}
            for k, v in ground_truth.items():
                if k in type_mapping.values():
                    for att in v:
                        true_labels[att] = k
            for config in eval_configs:
                current_config = deepcopy(config)
                current_config.model_dir = current_config.model_dir + str(run) + '/'
                current_config.exclude_data_origin.append(key)
                if CAiSE_VERSION:
                    current_config.exclude_data_origin.append(SCHEMA_ORG)
                aug_log, att_class, pred_labels = do_basic_extraction(current_config,value,key)
                taken1 = set()
                taken2 = set()
                for att, role in aug_log.attribute_to_concept_type.items():
                    if "case" not in att.lower():
                        atts.append(att)
                        atts_per_log.append(att)
                        atts_per_log_count[key] += 1
                    else:
                        continue
                    if aug_log.attribute_to_attribute_type[att] in [AttributeType.CASE_ATT, AttributeType.CASE, AttributeType.TIMESTAMP, AttributeType.NUMERIC] and att not in ["lifecycle:transition"]:
                        atts_handles_per_step["step_1"] += 1
                        atts_per_log_handled_per_step[key]["step_1"].append(att)
                        taken1.add(att)
                        write.writerow([key, att, "D", "D"])
                    elif aug_log.attribute_to_attribute_type[att] == AttributeType.RICH_TEXT:# or aug_log.attribute_to_attribute_type[att] == AttributeType.TEXT:
                        atts_handles_per_step["step_2"] += 1
                        atts_per_log_handled_per_step[key]["step_2"].append(att)
                        taken2.add(att)
                        write.writerow([key, att, "D_T", "D_T"])
                    elif aug_log.attribute_to_attribute_type[att] != AttributeType.RICH_TEXT and att in true_labels.keys() and true_labels[att] in [type_mapping[ConceptType.BO_STATUS.value], type_mapping[ConceptType.ACTOR_INSTANCE.value], type_mapping[ConceptType.ACTOR_NAME.value], type_mapping[ConceptType.BO_NAME.value], type_mapping[ConceptType.ACTION_STATUS.value]]:
                        atts_handles_per_step["step_3_cons"] += 1
                        atts_per_log_handled_per_step[key]["step_3_cons"].append(att)
                    else:
                        atts_handles_per_step["step_3_other"] += 1
                        atts_per_log_handled_per_step[key]["step_3_other"].append(att)
                true_count = {}

                #print(pred_labels)

                pred = []
                true = []
                for att, p in pred_labels.items():
                    if att in taken1 or att in taken2:
                        continue
                    if att in true_labels.keys():# and att in [type_mapping[ConceptType.BO_STATUS.value],  type_mapping[ConceptType.ACTOR_INSTANCE.value], type_mapping[ConceptType.ACTOR_NAME.value], type_mapping[ConceptType.BO_NAME.value], type_mapping[ConceptType.ACTION_STATUS.value]]:
                        atts_per_label[p[0]].add(att)
                        atts_per_label_per_log[key][p[0]].add(att)
                        pred.append(p[0])
                        true.append(true_labels[att])
                        if att in taken1 or att in taken2:
                            continue
                        write.writerow([key, att, true_labels[att], p[0]])
                    else:
                        atts_per_label[p[0]].add(att)
                        atts_per_label_per_log[key][p[0]].add(att)

                        pred.append(p[0])
                        true.append('X')
                        #print("pred", att, p[0])
                        if att in taken1 or att in taken2:
                            continue
                        write.writerow([key, att, 'X', p[0]])
                for att, role in true_labels.items():
                    if att not in pred_labels.keys():
                        #print("att not in log: ", att)
                        pass
                    else:
                        if role not in true_count:
                            true_count[role] =1
                        true_count[role] += 1
                        if role not in true_count_all:
                            true_count_all[role] =1
                        true_count_all[role] += 1
                #         #if role in [type_mapping[ConceptType.BO_STATUS.value],  type_mapping[ConceptType.ACTOR_INSTANCE.value], type_mapping[ConceptType.BO_NAME.value], type_mapping[ConceptType.ACTION_STATUS.value]]:
                #         pred.append('X')
                #         true.append(true_labels[att])
                #         if att in taken1 or att in taken2:
                #             continue
                #         write.writerow([key, att, true_labels[att], 'X'])
                if len(pred) == 0:
                    continue
                # print(true_count)
                per_log_report = classification_report(true, pred, output_dict=True)
                for label, res in per_log_report.items():
                    #print(label, res)
                    if label == "accuracy":
                        if "accuracy" not in macro_dict.keys():
                            macro_dict["accuracy"] = []
                        macro_dict["accuracy"].append(res)
                        continue
                    for metric, val in res.items():
                        if metric not in macro_dict[label].keys():
                            macro_dict[label][metric] = []
                        macro_dict[label][metric].append(val)

                overall_pred.extend(pred)
                overall_true.extend(true)
        # results[key] = kb_class, bert_class, combined_bert, final, ld.samples
    if len(overall_true) == 0 or len(overall_pred) == 0:
        print("No results available!")
        return
    #print(atts)
    #print(len(atts))
    #print(atts_handles_per_step)
    #print(atts_per_log_count)
    #print(atts_per_log_handled_per_step)
    print(classification_report(overall_true, overall_pred))
    #print(overall_true)
    #print(overall_pred)
    # print(macro_dict)
    #print(type_mapping.values())
    #print(conf_matrix(overall_true, overall_pred, list(type_mapping.values())+['X']))
    print(atts_per_label)
    #print(true_count_all)


def evaluate_action_classification(eval_configs):
    overall_pred = []
    overall_true = []
    macro_dict = {label: {} for label in ACTION_IDX_TO_LABEL.values()}
    macro_dict["X"] = {}
    macro_dict["macro avg"] = {}
    macro_dict["weighted avg"] = {}
    actions_per_label = {label: set() for label in ACTION_IDX_TO_LABEL.values()}
    misclassified = {label: set() for label in ACTION_IDX_TO_LABEL.values()}

    for key, value in get_file_names(gold=False).items():
        run = tagged_logs_raw.index(key)
        ground_truth = loader.get_action_ground_truth(DEFAULT_RES_DIR, file="gt_actions.json")

        for config in eval_configs:
            current_config = deepcopy(config)
            current_config.model_dir = current_config.model_dir + str(run) + '/'
            current_config.exclude_data_origin.append(key)
            aug_log, word_embeddings = do_basic_extraction(current_config, value, key, att_cls=False)

            action_classifier = ActionClassifier(current_config, aug_log=aug_log, embeddings=word_embeddings)
            true_labels = {}
            for k, v in ground_truth.items():
                if k in action_classifier.actions:
                    true_labels[k] = v
            pred_labels = action_classifier.classify_actions()
            pred = []
            true = []
            for att, p in pred_labels.items():
                if att in true_labels.keys():
                    if p != true_labels[att]:
                        misclassified[p].add(att)
                    actions_per_label[p].add(att)
                    pred.append(p)
                    true.append(true_labels[att])
                else:
                    pass
                    # print("pred", att, p)
            if len(pred) == 0:
                continue
            per_log_report = classification_report(true, pred, output_dict=True)
            for label, res in per_log_report.items():
                #print(label, res)
                if label == "accuracy":
                    if "accuracy" not in macro_dict.keys():
                        macro_dict["accuracy"] = []
                    macro_dict["accuracy"].append(res)
                    continue
                for metric, val in res.items():
                    if metric not in macro_dict[label].keys():
                        macro_dict[label][metric] = []
                    macro_dict[label][metric].append(val)

            overall_pred.extend(pred)
            overall_true.extend(true)
        # results[key] = kb_class, bert_class, combined_bert, final, ld.samples
    if len(overall_true) == 0 or len(overall_pred) == 0:
        print("No results available!")
        return
    print(classification_report(overall_true, overall_pred))
    # print(overall_true)
    # print(overall_pred)
    # print(macro_dict)
    # print(list(ACTION_IDX_TO_LABEL.values()))
    # print(conf_matrix(overall_true, overall_pred, list(ACTION_IDX_TO_LABEL.values()) + ['X']))


def evaluate_action_classification_all_at_once(eval_configs):
    overall_pred = []
    overall_true = []
    macro_dict = {label: {} for label in ACTION_IDX_TO_LABEL.values()}
    macro_dict["X"] = {}
    macro_dict["macro avg"] = {}
    macro_dict["weighted avg"] = {}
    actions_per_label = {label: set() for label in ACTION_IDX_TO_LABEL.values()}
    misclassified = {label: set() for label in ACTION_IDX_TO_LABEL.values()}
    ground_truth = loader.get_action_ground_truth(DEFAULT_RES_DIR, file="gt_actions.json")
    for config in eval_configs:
        current_config = deepcopy(config)
        actions = list(ground_truth.keys())
        word_embeddings = WordEmbeddings(config=current_config)
        action_classifier = ActionClassifier(current_config, actions=actions, embeddings=word_embeddings)
        true_labels = {}
        for k, v in ground_truth.items():
            if k in action_classifier.actions:
                true_labels[k] = v
        pred_labels = action_classifier.classify_actions()
        pred = []
        true = []
        for att, p in pred_labels.items():
            if att in true_labels.keys():
                if p != true_labels[att]:
                    misclassified[p].add(att)
                actions_per_label[p].add(att)
                if p is not None and p != "None":
                    pred.append(p)
                    true.append(true_labels[att])
            else:
                pass
                # print("pred", att, p)
        if len(pred) == 0:
            continue
        per_log_report = classification_report(true, pred, output_dict=True)
        for label, res in per_log_report.items():
            #print(label, res)
            if label == "accuracy":
                if "accuracy" not in macro_dict.keys():
                    macro_dict["accuracy"] = []
                macro_dict["accuracy"].append(res)
                continue
            for metric, val in res.items():
                if metric not in macro_dict[label].keys():
                    macro_dict[label][metric] = []
                macro_dict[label][metric].append(val)

        overall_pred.extend(pred)
        overall_true.extend(true)
    if len(overall_true) == 0 or len(overall_pred) == 0:
        print("No results available!")
        return
    print(classification_report(overall_true, overall_pred))

confs = [
    DEFAULT_CONFIG
]

baseline_confs = [
    BASELINE_CONFIG
]

# UNCOMMENT THE EVALUATION OF INTEREST
if __name__ == '__main__':
    main_tic = time.perf_counter()
    #eval_overall_bert_performance(confs)
    #eval_against_baseline(baseline_confs)
    #eval_attribute_classification(eval_configs=confs)
    #evaluate_resource_classifier(eval_configs=confs)
    evaluate_action_classification_all_at_once(eval_configs=confs)
    main_toc = time.perf_counter()
    print(f"Program finished all operations in {main_toc - main_tic:0.4f} seconds")
    sys.exit()
