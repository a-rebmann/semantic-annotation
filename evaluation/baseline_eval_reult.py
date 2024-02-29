import evaluation.metrics as m


class BaselineEvalResult:

    def __init__(self, ground_truth, hmm, bert, allowed_tags):
        self.ground_truth = ground_truth
        self.hmm = hmm
        self.bert = bert
        self.allowed_tags = allowed_tags
        self.hmm_entity_level_f1 = m.f1_entity_level(self.ground_truth.values(), self.hmm.values())
        self.bert_entity_level_f1 = m.f1_entity_level(self.ground_truth.values(), self.bert.values())
        self.hmm_entity_level_prec = m.prec_entity_level(self.ground_truth.values(), self.hmm.values())
        self.bert_entity_level_prec = m.prec_entity_level(self.ground_truth.values(), self.bert.values())
        self.hmm_entity_level_rec = m.rec_entity_level(self.ground_truth.values(), self.hmm.values())
        self.bert_entity_level_rec = m.rec_entity_level(self.ground_truth.values(), self.bert.values())
        self.hmm_token_level_f1 = m.f1_token_level(self.ground_truth.values(), self.hmm.values())
        self.bert_token_level_f1 = m.f1_token_level(self.ground_truth.values(), self.bert.values())
        self.hmm_per_class_metrics, self.available_hmm_tags = m.per_class_metrics(self.ground_truth.values(), self.hmm.values())
        self.bert_per_class_metrics, self.available_bert_tags = m.per_class_metrics(self.ground_truth.values(), self.bert.values())
        self.bert_conf_matrix = m.conf_matrix(self.ground_truth.values(), self.bert.values())
        self.hmm_conf_matrix = m.conf_matrix(self.ground_truth.values(), self.hmm.values())

    def print_summary(self, name=None):
        if name:
            print('Evaluation summary for '+name)
        print('HMM parser: entity-level f1 score:' + str(self.hmm_entity_level_f1))
        print(self.hmm_per_class_metrics)
        print('-' * 40)
        print('BERT parser: entity-level f1 score:' + str(self.bert_entity_level_f1))
        print(self.bert_per_class_metrics)

    def print_examples(self, num_examples=20):
        i = 0
        for key in self.ground_truth.keys():
            true_tags = self.ground_truth[key]
            bert_tags = self.bert[key]
            hmm_tags = self.hmm[key]
            if bert_tags == hmm_tags:
                print(key + " | " + str(true_tags) + " | " + str(bert_tags) + " | " + str(hmm_tags))
                i += 1
            if i > num_examples:
                break
        print(40*'-')
        i = 0
        for key in self.ground_truth.keys():
            true_tags = self.ground_truth[key]
            bert_tags = self.bert[key]
            hmm_tags = self.hmm[key]
            if bert_tags != hmm_tags:
                print(key + " | " + str(true_tags) + " | " + str(bert_tags) + " | " + str(hmm_tags))
                i += 1
            if i > num_examples:
                break
