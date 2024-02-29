import json

from nltk import WordNetLemmatizer

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from const import ConceptType, ACTION_IDX_TO_LABEL
import operator
from nltk.corpus import words



class ActionClassifier:

    def __init__(self, config, embeddings, aug_log=None, actions=None):
        self.config = config
        self.aug_log = aug_log
        if actions is not None:
            self.actions = list([act for act in actions if any(tok in words.words() for tok in act.split(" "))])
        elif self.aug_log is not None:
            self.actions = list([act for act in self.aug_log.get_all_unique_values_for_role(ConceptType.ACTION_NAME.value) if any(tok in words.words() for tok in act.split(" "))])
        self.embeddings = embeddings
        self.lemmatizer = WordNetLemmatizer()
        with open(self.config.resource_dir + 'mitphb.json') as json_file:
            action_taxonomy = json.load(json_file)
        # all unique actions
        unique_actions_taxonomy = set()
        # a mapping from all unique actions to their top most ancestor(s)
        child_to_upper_level = dict()
        # all upper level actions
        upper_acts = set()
        self.unique_actions_from_taxonomy(action_taxonomy, unique_actions_taxonomy, child_to_upper_level,
                                          upper_acts)
        self.unique_actions_taxonomy = unique_actions_taxonomy
        self.child_to_upper_level=child_to_upper_level
        self.upper_acts = upper_acts

    def classify_actions(self):
        #self.produce_gs()
        #self.build_classifier()
        return {act: self.get_action_type_for_action(act) for act in self.actions}

    def unique_actions_from_taxonomy(self, action_taxonomy, unique_actions, child_to_upper_level, upper_acts, upper_level=None):
        for act, children in action_taxonomy.items():
            unique_actions.add(act)
            if upper_level is None:
                child_to_upper_level[act] = {act}
                upper_acts.add(act)
                ul = act
            else:
                if act in child_to_upper_level:
                    child_to_upper_level[act].add(upper_level)
                else:
                    child_to_upper_level[act] = {upper_level}
                ul = upper_level
            for child in children:
                self.unique_actions_from_taxonomy(child, unique_actions, child_to_upper_level, upper_acts, upper_level=ul)


    def produce_gs(self):
        with open(self.config.resource_dir + 'gt_actions.json') as json_file:
            gt = json.load(json_file)
            with open(self.config.resource_dir + 'mitphb.json') as json_file:
                action_taxonomy = json.load(json_file)
                # all unique actions
                unique_actions_taxonomy = set()
                # a mapping from all unique actions to their top most ancestor(s)
                child_to_upper_level = dict()
                # all upper level actions
                upper_acts = set()
                self.unique_actions_from_taxonomy(action_taxonomy, unique_actions_taxonomy, child_to_upper_level, upper_acts)
                #print(unique_actions_taxonomy)
                #print(child_to_upper_level)
                #print(upper_acts)
                for action in self.actions:
                    ms = self.get_most_similar(action, unique_actions_taxonomy, child_to_upper_level, upper_acts)
                    if action in gt:
                        continue
                    else:
                        gt[action] = ms
            with open(self.config.resource_dir + 'gt_actions.json', 'w') as outfile:
                json.dump(gt, outfile)

    def get_most_similar(self, action, taxonomy_actions, child_to_upper_level, upper_acts):
        if len(action) < 3:
            return "None"
        sims = {}
        upper_level_sims = {}
        #combined_sims = {}
        for tax_action in taxonomy_actions:
            try:
                sim = self.embeddings.embeddings.similarity(action, tax_action)
                #print(action, tax_action, sim)
                if tax_action in upper_acts:
                    upper_level_sims[tax_action] = sim
                sims[tax_action] = sim
            except KeyError as e:
                #print(e)
                action = self.lemmatizer.lemmatize((action.split(" ")[-1]))
                try:
                    sim = self.embeddings.embeddings.similarity(action, tax_action)
                    #print(action, tax_action, sim)
                    if tax_action in upper_acts:
                        upper_level_sims[tax_action] = sim
                    sims[tax_action] = sim
                except KeyError as e:
                    pass
                    #print(e, "after lemmatization still")
        if len(sims) == 0:
            return "None"

        # for u_act, u_sim in upper_level_sims.items():
        #     for act, sim in sims.items():
        #         if u_act in self.child_to_upper_level[act]:
        #             combined_sims[(u_act, act)] = u_sim + sim

        max_sim = max(sims.items(), key=operator.itemgetter(1))[0]
        max_sim_upper = max(upper_level_sims.items(), key=operator.itemgetter(1))[0]
        max_sim_upper_ini = str(max_sim_upper)
        #max_sim_combined = max(combined_sims.items(), key=operator.itemgetter(1))[0][0]

        #print("MAX any",  action, max_sim, sims[max_sim])
        #print("MAX upper",  action, max_sim_upper, sims[max_sim_upper])
        #print("MAX combi", action, max_sim_combined, sims[max_sim_combined])

        #if sims[max_sim] <= upper_level_sims[max_sim_upper_ini]+0.05:
        #    max_sim = max_sim_upper_ini

        if len(child_to_upper_level[max_sim]) == 1:
            max_sim = list(child_to_upper_level[max_sim])[0]
        else:
            max_sim_upper = -1
            for upper_level_act in child_to_upper_level[max_sim]:
                if upper_level_sims[upper_level_act] > max_sim_upper:
                    max_sim = upper_level_act
                    max_sim_upper = upper_level_sims[upper_level_act]

        #print("MAX top-level", action, max_sim, sims[max_sim])

        #if sims[max_sim_any] < .5:
        #    return "None"
        return max_sim if sims[max_sim] > 0 else max_sim_upper_ini

    def get_action_type_for_action(self, action):
        return self.get_most_similar(action, self.unique_actions_taxonomy,self.child_to_upper_level, self.upper_acts)

    def build_classifier(self):
        with open(self.config.resource_dir + 'mitphb.json') as json_file:
            action_taxonomy = json.load(json_file)
        # all unique actions
        unique_actions_from_taxonomy = set()
        # a mapping from all unique actions to their top most ancestor(s)
        child_to_upper_level = dict()
        # all upper level actions
        upper_acts = set()
        self.unique_actions_from_taxonomy(action_taxonomy, unique_actions_from_taxonomy, child_to_upper_level,
                                          upper_acts)

        # define documents
        docs = [act for act in unique_actions_from_taxonomy]
        # define class labels
        label_2_idx = {lab: idx for idx, lab in ACTION_IDX_TO_LABEL.items()}
        labels = array([label_2_idx[child_to_upper_level[doc].pop()] for doc in docs])
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
            if word in self.embeddings.embeddings:
                embedding_vector = self.embeddings.embeddings[word]
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



