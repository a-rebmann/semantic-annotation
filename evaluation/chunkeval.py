from seqeval.metrics import f1_score, classification_report


class ChunkEval:

    def __init__(self, ground_truth:dict, pred):
        self.ground_truth = ground_truth
        self.pred = pred
        self.available_tags = []
        self.bert_per_class_metrics = self.prepare_chunks()

    def prepare_chunks(self):
        res = {'BO': {}, 'A': {}, 'ACTOR': {}, 'REC': {}, 'ASTATE': {}, 'BOSTATE': {}, 'X': {}}
        iob_gt = {'BO': {}, 'A': {}, 'ACTOR': {}, 'REC': {}, 'ASTATE': {}, 'BOSTATE': {}, 'X': {}}
        iob_pred = {'BO': {}, 'A': {}, 'ACTOR': {}, 'REC': {}, 'ASTATE': {}, 'BOSTATE': {}, 'X': {}}
        for ent_type in iob_gt.keys():
            for key, value in self.ground_truth.items():
                new_tags = []
                for tag in value:
                    if tag != ent_type:
                        new_tags.append('O')
                    else:
                        new_tags.append('I-MISC')
                iob_gt[ent_type][key] = new_tags
            for key, value in self.pred.items():
                new_tags = []
                for tag in value:
                    if tag != ent_type:
                        new_tags.append('O')
                    else:
                        new_tags.append('I-MISC')
                iob_pred[ent_type][key] = new_tags
            print(iob_pred)
            print(iob_gt)
            cls_rep = classification_report(iob_gt[ent_type].values(), iob_pred[ent_type].values(), output_dict=True)
            print(ent_type)
            print(cls_rep)
            if 'MISC' in cls_rep.keys():
                self.available_tags.append(ent_type)
                res[ent_type]["support"] = cls_rep['MISC']["support"]
                res[ent_type]["f1-score"] = cls_rep['MISC']["f1-score"]
                res[ent_type]["precision"] = cls_rep['MISC']["precision"]
                res[ent_type]["recall"] = cls_rep['MISC']["recall"]
        return res

    def print_summary(self, name=None):
        if name:
            print('-'*40)
            print('Evaluation summary for '+name)
        print('BERT parser: entity-level f1 score:' + str(self.bert_entity_level_f1))
        print(self.bert_per_class_metrics)

    def print_examples(self, num_examples=20):
        i = 0
        for key in self.ground_truth.keys():
            true_tags = self.ground_truth[key]
            bert_tags = self.bert[key]
            print(key + " | " + str(true_tags) + " | " + str(bert_tags))
            i += 1
            if i > num_examples:
                break

