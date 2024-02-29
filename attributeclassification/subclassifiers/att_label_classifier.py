from sklearn import feature_extraction, feature_selection
from pandas import DataFrame

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from const import consider_for_label_classification, class_labels, ConceptType, act
from data.labeled_data import LabeledData
from model.augmented_log import AugmentedLog
from preprocessing.preprocessor import clean_attribute_name
from sklearn.linear_model import SGDClassifier
from nltk import word_tokenize
import numpy as np
from readwrite.loader import deserialize_model
from readwrite.writer import serialize_model
from nltk.corpus import words
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

CAiSE_VERSION = False

def tfidf_glove(idf_dict, vals, glove):
    vectors = []
    for idx, val in enumerate(vals):
        glove_vectors = [glove[tok] if tok in glove.key_to_index.keys() else np.zeros(50) for tok in word_tokenize(val)]
        weights = [idf_dict.get(word, 1) if word in glove.key_to_index.keys() else 0.0000000001 for word in
                   word_tokenize(val)]
        try:
            vectors.append(np.average(glove_vectors, axis=0, weights=weights))
        except ZeroDivisionError:
            #print(val, "caused Zero Division error")
            vectors.append(np.zeros(50))
    return np.array(vectors)


class AttributeLabelClassifier:

    def __init__(self, config, data: LabeledData, aug_log: AugmentedLog, embeddings):
        docs, concepts = data.get_attribute_data()
        #print(len(docs), len(concepts))
        docs = [doc.lower() for i, doc in enumerate(docs) if concepts[i] in class_labels]
        docs = [self.pp_doc(doc) for doc in docs]

        concepts = [conc for conc in concepts if conc in class_labels]

        #print([(doc, con) for doc, con in zip(docs, concepts)])
        #print(len(docs), len(concepts))
        self.config = config
        self.d = DataFrame({"text": docs, "y": concepts})
        self.cols = aug_log.get_attributes_by_att_types(consider_for_label_classification)
        self.aug_log = aug_log
        self.embeddings = embeddings

    def with_tf_idf_and_embedding(self, eval_mode=False):
        # self.build_classifier()
        res = {}
        md = deserialize_model(self.config.resource_dir, "att_class")

        if md is False or eval_mode is True:
            print("Build new attribute classifier")
            tfidf = feature_extraction.text.TfidfVectorizer(ngram_range=(1, 2))
            tfidf.fit_transform(self.d["text"].values)
            # test = self.test["text"].values
            # Now lets create a dict so that for every word in the corpus we have a corresponding IDF value
            idf_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
            x_train = tfidf_glove(idf_dict, self.d["text"].values, self.embeddings)
            # x_test = tfidf_glove(idf_dict, test, glove)
            enc = LabelEncoder()
            X = class_labels
            enc.fit(X)

            y_train = enc.transform(self.d["y"].values)
            # y_test = enc.transform(self.test["y"].values)
            clzz = build_log_reg(x_train, y_train)
            model_dict = {"idf_dict": idf_dict, "X": X, "enc": enc, "clzz": clzz}
            serialize_model(self.config.resource_dir, model_dict, "att_class")
        else:
            idf_dict = md["idf_dict"]
            enc = md["enc"]
            clzz = md["clzz"]
            X = md["X"]

        for plain in self.cols:
            #if not CAiSE_VERSION:
            clean = self.prepare_name(plain)
            clean = self.pp_doc(clean)
            #else:
            #    clean = plain
            if clean.strip() == "" or clean.strip() in act:
                #print(plain, "X", 1)
                res[plain] = "X", 1
            # elif clean in self.aug_log.get_all_unique_values_for_role(ConceptType.BO_NAME.value):
            #     print("found", clean, "in object types")
            #     res[plain] = "BO", 1
            elif any(self.check_proper_word(tok) for tok in clean.split(" ")):
                x_t = tfidf_glove(idf_dict, [clean], self.embeddings)
                probas = clzz.predict_proba(x_t)[0]
                pred = enc.inverse_transform(clzz.predict(x_t))[0]
                #print(clean, plain, pred, probas[X.index(pred)])
                res[plain] = pred, probas[X.index(pred)]
            else:
                #print(plain, "X", 1)
                res[plain] = "X", 1

        # run_log_reg(x_train, x_test, y_train, y_test, enc)
        return res

    def check_proper_word(self, tok):
        return len(tok) > 2 and tok in words.words() or "type" in tok or "id" in tok

    def pp_doc(self, doc):
        return doc.replace("type", "").replace("uuid", "").replace("identity", "").replace("id", "")

    def build_classifier(self):
        res = {}
        docs = [act for act in self.d["text"].values]
        # define class labels
        label_2_idx = {lab: idx for idx, lab in enumerate(self.d["y"].unique())}  # TODO
        print(label_2_idx)
        idx_2_label = {idx: lab for lab, idx in label_2_idx.items()}  # TODO
        labels = array([label_2_idx[lab] for lab in self.d["y"].values])  # TODO
        print(docs)
        print(labels)
        # prepare tokenizer
        t = Tokenizer()
        t.fit_on_texts(docs)
        vocab_size = len(t.word_index) + 1
        # integer encode the documents
        encoded_docs = t.texts_to_sequences(docs)
        print(encoded_docs)
        # pad documents to a max length of 4 words
        max_length = 2
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
        print(padded_docs)
        # create a weight matrix for words in training docs
        embedding_matrix = zeros((vocab_size, 50))
        for word, i in t.word_index.items():
            if word in self.embeddings:
                embedding_vector = self.embeddings[word]
                embedding_matrix[i] = embedding_vector
        # define model
        model = Sequential()
        e = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=2, trainable=False)
        model.add(e)
        model.add(Flatten())
        model.add(Dense(1, activation='softmax'))
        # compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # summarize the model
        print(model.summary())
        # fit the model
        model.fit(padded_docs, labels, epochs=50, verbose=0)
        # evaluate the model
        loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
        print('Accuracy: %f' % (accuracy * 100))

        for plain in self.cols:
            clean = self.prepare_name(plain)
            if any(self.check_proper_word(tok) for tok in clean.split(" ")):
                encoded_clean = t.texts_to_sequences([clean])
                padded_clean = pad_sequences(encoded_clean, maxlen=max_length, padding='post')
                print(clean)
                print(model.predict(padded_clean))

    def prepare_name(self, plain):
        return clean_attribute_name(plain).replace("doc", "document").replace("type", "").replace("uuid", "").replace(
            "identity", "").replace("id", "")


def build_log_reg(train_features, y_train, alpha=1e-4):
    log_reg = SGDClassifier(loss='log', alpha=alpha, n_jobs=-1, penalty='l2')
    log_reg.fit(train_features, y_train)
    return log_reg
