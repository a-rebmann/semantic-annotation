from config import Config
from const import EVAL_MODE, MatchingMode
from data.gathering.preprocess_bpmai import extract_service_tasks_from_jsons
from evaluate import EVAL_INPUT_DIR, BERT_DIR_ON_MACHINE, EVAL_OUTPUT_DIR
from main import DEFAULT_RES_DIR

DEFAULT_CONFIG = Config(input_dir=EVAL_INPUT_DIR, model_dir=BERT_DIR_ON_MACHINE, resource_dir=DEFAULT_RES_DIR,
                        output_dir=EVAL_OUTPUT_DIR, bo_ratio=0.5, conf_thresh=0.5, instance_thresh=0.5,
                        exclude_data_origin=[], we_file="glove-wiki-gigaword-50", matching_mode=MatchingMode.EXACT,
                        mode=EVAL_MODE, res_type=True)

if __name__ == '__main__':
    extract_service_tasks_from_jsons(DEFAULT_CONFIG)