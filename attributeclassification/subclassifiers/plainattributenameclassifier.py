from numpy import asarray, array
from numpy import zeros
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding

from preprocessing.preprocessor import preprocess_label

from data.gathering.schemaorgextraction import read_and_extract


from main import DEFAULT_RES_DIR


def train_and_classify():
    actor_terms, act_terms, action_status_terms, obj_terms, obj_status_terms = read_and_extract()
    docs = []
    labels = []
    for doc in actor_terms:
        docs.append(preprocess_label(doc))
        labels.append(0)
    # for doc in act_terms:
    #     docs.append(doc)
    #     labels.append(1)
    # for doc in action_status_terms:
    #     docs.append(doc)
    #     labels.append(2)
    for doc in obj_terms:
        docs.append(preprocess_label(doc))
        labels.append(1)
    # for doc in obj_status_terms:
    #     docs.append(preprocess_label(doc))
    #     labels.append(2)
    # prepare tokenizer
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
    f = open('../' + DEFAULT_RES_DIR + '/glove.6B.100d.txt')
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


if __name__ == '__main__':
    train_and_classify(DEFAULT_RES_DIR)