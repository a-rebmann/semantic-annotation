import evaluation.metrics as m


class BertEvalResult:
    def __init__(self, ground_truth, bert):
        self.ground_truth = ground_truth
        self.bert = bert
        self.bert_entitiy_level_support = len(self.ground_truth.values())
        self.bert_entity_level_f1 = m.f1_entity_level(self.ground_truth.values(), self.bert.values())
        self.bert_entity_level_prec = m.prec_entity_level(self.ground_truth.values(), self.bert.values())
        self.bert_entity_level_rec = m.rec_entity_level(self.ground_truth.values(), self.bert.values())
        self.bert_per_class_metrics, self.available_tags = m.per_class_metrics(self.ground_truth.values(), self.bert.values())
        self.conf_matrix = m.conf_matrix(self.ground_truth.values(), self.bert.values())

    def print_summary(self, name=None):
        if name:
            print('-'*40)
            print('Evaluation summary for '+name)
        print('BERT parser: entity-level f1 score:' + str(self.bert_entity_level_f1))
        print(self.bert_per_class_metrics)
        print('-'*40)
        print(self.conf_matrix)

    def print_examples(self, num_examples=20):
        i = 0
        for key in self.ground_truth.keys():
            true_tags = self.ground_truth[key]
            bert_tags = self.bert[key]
            if True: #bert_tags != true_tags:
                print(key + " | " + str(true_tags) + " | " + str(bert_tags))
                i += 1
            if i > num_examples:
                break
