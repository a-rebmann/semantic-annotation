import json
from collections import Counter
from datetime import timedelta

from nltk.corpus.reader import WordNetError
from numpy import asarray, array
from numpy import zeros
from simpletransformers.classification import ClassificationModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
import pandas as pd
import numpy as np

from nltk.corpus import wordnet as wn

from const import ConceptType, RESOURCE_IDX_TO_LABEL, TERMS_FOR_MISSING, AttributeType, type_mapping
from model.augmented_log import AugmentedLog
from preprocessing.nlp_utils import get_named_entities_if_present
from preprocessing.preprocessor import preprocess_label, clean_attribute_name, check_for_uuid
from data.gathering.schemaorgextraction import read_and_extract

from nltk import WordNetLemmatizer
from nltk.corpus import words
from data.gathering.conceptnet import get_json

system_keywords = [wn.synset("computer.n.01"), wn.synset("information.n.01"), wn.synset("system.n.01")]
human_keywords = [wn.synset("organization.n.01"), wn.synset("person.n.01"), wn.synset('physical_entity.n.01')]


def check_instance_basic(res):
    return any(char.isdigit() for char in str(res)) or check_for_uuid(str(res))


class ResourceClassifier:

    def __init__(self, config, aug_log: AugmentedLog = None):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.aug_log = aug_log
        self.model = ClassificationModel("bert", self.config.resource_dir + "resource_bert", use_cuda=False)

    def classify_resources(self):
        actor_instance_cols = []
        actor_name_cols = []
        # pass_cols = []
        for column in self.aug_log.augmented_df.columns:
            if len(self.aug_log.augmented_df[column].unique()) > 1000:
                continue

            if type_mapping[ConceptType.ACTOR_NAME.value] in column and ":instance" not in column:
                if ":a:" in column:
                    actor_name_cols.append((column, len(self.aug_log.augmented_df[column].unique())))
                else:
                    actor_instance_cols.append(column)
            elif type_mapping[ConceptType.ACTOR_INSTANCE.value] in column:
                if ":a:" in column:
                    actor_name_cols.append((column, len(self.aug_log.augmented_df[column].unique())))
                else:
                    actor_instance_cols.append(column)
        actor_name_cols.sort(key=lambda x: x[1], reverse=True)
        actor_name_cols = [col[0] for col in actor_name_cols]
        if "org:resource" in self.aug_log.augmented_df.columns:
            actor_name_cols.insert(0, "org:resource")
        elif "org:group" in self.aug_log.augmented_df.columns:
            actor_name_cols.insert(0, "org:group")
        elif "org:role" in self.aug_log.augmented_df.columns:
            actor_name_cols.insert(0, "org:role")
        self.aug_log.augmented_df["unique_resource"] = pd.Series()
        # adjusted to first consider the first actor_instance as unique resources, then the actor_roles
        # OPTIONAL introduce intermediate column summarizing all unique actor_role combinations first
        for column in actor_instance_cols + actor_name_cols:  # + pass_cols:
            self.aug_log.augmented_df["unique_resource"] = np.where(self.aug_log.augmented_df["unique_resource"].isnull(),
                                                                  self.aug_log.augmented_df[column],
                                                                  self.aug_log.augmented_df["unique_resource"])
        self.aug_log.df["unique_resource"] = self.aug_log.augmented_df["unique_resource"]

        unique_main_resources = self.aug_log.augmented_df["unique_resource"].dropna().unique()
        unique_main_resources = [res for res in unique_main_resources if res not in TERMS_FOR_MISSING]

        clear_preds_res = {}
        clear_preds_res_text = {}
        clear_preds_res_misc = {}

        clear_preds_res_ne = {}
        clear_preds_res_wn = {}
        clear_preds_res_bert = {}
        clear_preds_res_other = {}
        #
        # we get te cleaned labels to feed to the classifier
        try:
            labels = list(self.aug_log.df[self.aug_log.event_label].unique())
        except KeyError:
            labels = list(self.aug_log.df[self.aug_log.event_label.lower()].unique())
            self.aug_log.event_label = self.aug_log.event_label.lower()

        cleaned = [preprocess_label(label) for label in labels if preprocess_label(label) not in TERMS_FOR_MISSING]
        # print(cleaned)
        bo_and_action_ratio = sum(
            [1 if 'A' in self.aug_log.tagged_labels[lab][1] and 'BO' in self.aug_log.tagged_labels[lab][1] else 0 for
             lab in cleaned]) / len(cleaned)
        cleaned_to_label = {clean: lab for clean, lab in zip(cleaned, labels)}

        label_for_to_predict = {str(res): label for res in unique_main_resources for label in self.aug_log.tagged_labels
                                if preprocess_label(res) in label}

        # print(len(unique_main_resources), "resources to be classified")
        # for res in unique_main_resources:
        #      print("\"" + res + "\",")
        labels_per_res = self.get_labels_per_resource("unique_resource")
        #resources_per_label = self.get_resources_per_label("unique_resource")

        #
        # we predict the label for each activity (classifier is trained on activity labels tagged with either HUM or SYS
        predictions, raw_outputs = self.model.predict(cleaned)
        predictions = [RESOURCE_IDX_TO_LABEL[pred] for pred in predictions]

        label_predictions = {r: pred for r, pred in zip(cleaned, predictions)}

        #
        # check if the time difference to the previous event is always 0 when a resource is present -> Potential System
        pot_sys_res = self.check_time_diffs("unique_resource")
        for res in pot_sys_res:
            if pd.isna(res) or res in TERMS_FOR_MISSING:
                continue
            if res not in clear_preds_res:
                clear_preds_res[res] = "SYS"
                clear_preds_res_misc[res] = "SYS"
                clear_preds_res_other[res] = "SYS"

        # NAMED ENTITIES
        count_hum = 0
        count_sys = 0
        for res in unique_main_resources:
            nes = get_named_entities_if_present(res)

            if len(nes) > 0:
                for entry in nes:
                    if entry[3] == 'PERSON' or entry[3] == 'ORG' or entry[3] == 'GPE' or entry[3] == 'LAW':
                        count_hum += 1

                        # clear_preds_res[res] = "HUM"
                    elif entry[3] == 'PRODUCT' or entry[3] == 'WORK_OF_ART':
                        count_sys += 1

                        # clear_preds_res[res] = "SYS"
        if count_hum > (len(labels_per_res.keys()) / 2) and len(actor_instance_cols) == 0:
            for res in labels_per_res.keys():
                # print(res, "NER", "HUM")
                clear_preds_res[res] = "HUM"
                clear_preds_res_ne[res] = "HUM"
                clear_preds_res_text[res] = "HUM"
        elif count_sys > (len(labels_per_res.keys()) / 2) and len(actor_instance_cols) == 0:
            for res in labels_per_res.keys():
                # print(res, "NER", "SYS")
                clear_preds_res[res] = "SYS"
                clear_preds_res_ne[res] = "SYS"
                clear_preds_res_text[res] = "SYS"

        excl = set()
        # EXCLUSIVE EXECUTION
        # for label, reses in resources_per_label.items():
        #
        #     # We check if this resource exclusively execute activities
        #     # (since we have instances of resources here the fact that a single instance executes an
        #     # activity exclusively is an indicator for a system)
        #     if len(reses) == 1:
        #         res = reses.pop()
        #         if res in clear_preds_res:
        #             continue
        #         if pd.isna(res) or res in TERMS_FOR_MISSING:
        #             continue
        #         if len(labels_per_res[res]) == 1 and res not in clear_preds_res:
        #             # clear_preds_res[res] = "SYS"
        #             excl.add(res)
        #             # print(label, res, "exclusive")

        already_predicted = {}
        checked = {}
        for res, labels in labels_per_res.items():
            #print(res)
            if pd.isna(res) or res in TERMS_FOR_MISSING:
                continue
            if res in clear_preds_res:
                print("Skipping", res)
                continue
            wn_results = {}
            preprocessed = preprocess_label(res)

            if preprocessed in checked:
                wn_results[preprocessed] = checked[preprocessed]
            elif not check_instance_basic(res) and all(
                    len(r) > 2 for r in preprocessed.split(" ")) and self.lemmatizer.lemmatize(
                    preprocessed) in words.words():  # preprocessed in label_for_to_predict and
                try:
                    lem = self.lemmatizer.lemmatize(preprocessed)
                    self.check_wordnet(preprocessed, wn.synset(lem + '.n.01'), wn_results)
                except WordNetError as e:
                    print(e)
            #print(wn_results)
            if preprocessed in wn_results:
                checked[preprocessed] = wn_results[preprocessed]
                clear_preds_res[res] = wn_results[preprocessed]
                clear_preds_res_text[res] = wn_results[preprocessed]
                clear_preds_res_wn[res] = wn_results[preprocessed]
            if res not in clear_preds_res:
                if res in already_predicted:
                    clear_preds_res[res] = already_predicted[res][0]
                    # print(res, already_predicted[res][0], already_predicted[res][1])
                    if already_predicted[res][1] == "TEXT":
                        clear_preds_res_text[res] = already_predicted[res][0]
                    else:
                        clear_preds_res_misc[res] = already_predicted[res][0]
                    if already_predicted[res][2] == "BERT":
                        clear_preds_res_bert[res] = already_predicted[res][0]
                    elif already_predicted[res][2] == "OTHER":
                        clear_preds_res_other[res] = already_predicted[res][0]
                elif preprocessed in already_predicted:
                    clear_preds_res[res] = already_predicted[preprocessed][0]
                    # print(res, already_predicted[preprocessed][0], already_predicted[preprocessed][1])
                    if already_predicted[preprocessed][1] == "TEXT":
                        clear_preds_res_text[res] = already_predicted[preprocessed][0]
                    else:
                        clear_preds_res_misc[res] = already_predicted[preprocessed][0]
                    if already_predicted[preprocessed][2] == "BERT":
                        clear_preds_res_bert[res] = already_predicted[preprocessed][0]
                    elif already_predicted[preprocessed][2] == "OTHER":
                        clear_preds_res_other[res] = already_predicted[preprocessed][0]
                elif res not in label_for_to_predict:
                    if bo_and_action_ratio < .4:
                        clear_preds_res[res] = "HUM"
                        clear_preds_res_misc[res] = "HUM"
                        clear_preds_res_other[res] = "HUM"
                        already_predicted[res] = ("HUM", "MISC", "OTHER")
                    else:
                        preds = {cleaned_to_label[lab]: pred for lab, pred in label_predictions.items() if
                                 cleaned_to_label[lab] in labels}
                        if len(preds) == 0:
                            continue
                        cnt = Counter(predictions)
                        if len(cnt) > 1:
                            value1, count1 = cnt.most_common(2)[0]
                            value2, count2 = cnt.most_common(2)[1]
                            if count1 == count2:
                                value = "HUM"
                            else:
                                value, count = cnt.most_common(1)[0]
                        else:
                            value, count = cnt.most_common(1)[0]
                        # print(res, value, "no text")
                        clear_preds_res[res] = value
                        clear_preds_res_misc[res] = value
                        clear_preds_res_bert[res] = value
                        already_predicted[res] = (value, "MISC", "BERT")
                elif preprocessed == label_for_to_predict[res] and any(self.lemmatizer.lemmatize(
                        pp) in words.words() for pp in preprocessed.split(" ")):
                    # self.lemmatizer.lemmatize(preprocessed) in words.words():

                    labs = labels_per_res[res]
                    if len(preprocessed.split(" ")) > 1 and all(len(tok) > 1 for tok in preprocessed.split(" ")):
                        to_pred = [preprocessed]
                    else:
                        to_pred = [lab + " by " + preprocessed for lab in labs]

                    # to_pred = [". ".join(to_pred)] #TODO experimental

                    predictions, raw_outputs = self.model.predict(to_pred)
                    if len(predictions) == 0:
                        continue
                    #
                    cnt = Counter(predictions)
                    if len(cnt) > 1:
                        value1, count1 = cnt.most_common(2)[0]
                        value2, count2 = cnt.most_common(2)[1]
                        if count1 == count2:
                            value = 1
                        else:
                            value, count = cnt.most_common(1)[0]
                    else:
                        value, count = cnt.most_common(1)[0]
                    # print(res, value, "resource text")
                    clear_preds_res[res] = RESOURCE_IDX_TO_LABEL[value]
                    clear_preds_res_misc[res] = RESOURCE_IDX_TO_LABEL[value]
                    clear_preds_res_bert[res] = RESOURCE_IDX_TO_LABEL[value]
                    already_predicted[preprocessed] = (RESOURCE_IDX_TO_LABEL[value], "MISC", "BERT")
                else:
                    predss = {cleaned_to_label[lab]: pred for lab, pred in label_predictions.items() if
                             cleaned_to_label[lab] in labels}
                    if len(predss) == 0:
                        continue
                    cnt = Counter(predss.values())
                    if len(cnt) > 1:
                        value1, count1 = cnt.most_common(2)[0]
                        value2, count2 = cnt.most_common(2)[1]
                        if count1 == count2:
                            value = "HUM"
                        else:
                            value, count = cnt.most_common(1)[0]
                    else:
                        value, count = cnt.most_common(1)[0]
                    # print(res, value, "resource from label")
                    clear_preds_res[res] = value
                    already_predicted[res] = (value, "TEXT", "BERT")
                    clear_preds_res_text[res] = value
                    clear_preds_res_bert[res] = value
        #print(clear_preds_res, clear_preds_res_text, clear_preds_res_misc, clear_preds_res_ne, clear_preds_res_wn, clear_preds_res_bert, clear_preds_res_other)
        return clear_preds_res, clear_preds_res_text, clear_preds_res_misc, clear_preds_res_ne, clear_preds_res_wn, clear_preds_res_bert, clear_preds_res_other

    def train_and_classify(self):
        actor_terms, _, _, _, _ = read_and_extract(self.config.resource_dir)
        docs = []
        labels = []
        for doc in actor_terms:
            docs.append(preprocess_label(doc))
            labels.append(0)

        labels = array(labels)
        t = Tokenizer()
        t.fit_on_texts(docs)
        vocab_size = len(t.word_index) + 1
        # integer encode the documents
        encoded_docs = t.texts_to_sequences(docs)
        print(encoded_docs)
        # pad documents to a max length of 4 words
        max_length = 4
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        print(padded_docs)
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(self.config.resource_dir + '/glove.6B.100d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        # create a weight matrix for words in training docs
        embedding_matrix = zeros((vocab_size, 100))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        # define .model
        model = Sequential()
        e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
        model.add(e)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        # compile the .model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # summarize the .model
        print(model.summary())
        # fit the .model
        model.fit(padded_docs, labels, epochs=50, verbose=0)
        # evaluate the .model
        loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
        print('Accuracy: %f' % (accuracy * 100))

    def check_wordnet(self, original, word, res, cnt=0):
        #print(word)
        if cnt > 10:
            return
        if word in human_keywords:
            res[original] = ConceptType.HUMAN_NAME.value
            return
        elif word in system_keywords:
            res[original] = ConceptType.SYSTEM_NAME.value
            return
        else:
            try:
                hypernyms = word.hypernyms()
            except WordNetError as e:
                #print(e)
                return
            #print(word, hypernyms)
            for hypernym in hypernyms:
                if hypernym != word:
                    self.check_wordnet(original, hypernym, res, cnt + 1)
        return

    def get_labels_per_resource(self, att):
        result = {}
        grouped = self.aug_log.df.groupby(att)
        for val, frame in grouped:
            result[str(val)] = set(frame[self.aug_log.event_label].unique())
        return result

    def get_resources_per_label(self, att):
        res = {}
        grouped = self.aug_log.df.groupby(self.aug_log.event_label)
        for val, frame in grouped:
            res[str(val)] = set(frame[att].unique())
        return res

    def check_time_diffs(self, att):
        if all(self.aug_log.df[self.aug_log.timestamp].dt.minute == 0) and all(
                self.aug_log.df[self.aug_log.timestamp].dt.second == 0):
            sorted_df = self.aug_log.df.sort_values([self.aug_log.case_id, self.aug_log.timestamp],
                                                    ascending=[True, True])
            pot_sys_res = set()
            uniques = self.aug_log.df[att].unique()
            sorted_df["diff"] = sorted_df[self.aug_log.timestamp].shift(1) - sorted_df[self.aug_log.timestamp]
            sorted_df["same_case"] = sorted_df[self.aug_log.case_id].shift(1) == sorted_df[self.aug_log.case_id]
            for u_res in uniques:
                # print(u_res)
                # & sorted["diff_case"]
                if (sorted_df.loc[sorted_df[att] == u_res]["same_case"] & (
                        sorted_df.loc[sorted_df[att] == u_res]["diff"] == timedelta(seconds=0))).all():
                    pot_sys_res.add(u_res)
            return pot_sys_res
        return set()
