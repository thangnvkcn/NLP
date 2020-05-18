from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout,concatenate,Embedding
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import  pickle
import os
sep  = os.sep
data_folder = "Data"
file_path = '../intentDetection/Data/data_intent_17.csv'
NUM_WORDS = 1500
def getData(file_path):
    data = pd.read_csv(file_path)
    return data
dic={}
d = getData(file_path)
intents = d.intent.unique()
def getLabels(data):

    texts = data.sentence
    for i,intent in enumerate(intents):
        dic[intent] = i
    labels = data.intent.apply(lambda x:dic[x])
    return texts,labels


def txtTokenizer(data):
    sentences=data.sentence
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    return tokenizer,word_index

if not os.path.exists(data_folder + sep +"data.pkl"):
    print("Data file not found, build it!")
    data = getData(file_path)
    texts, labels = getLabels(data)
    tokenizer, word_index = txtTokenizer(data)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X)

    y = to_categorical(np.asarray(labels[data.index]))
    file  =  open(data_folder + sep + "data.pkl", 'wb')
    pickle.dump([X,y,texts],file)
    file.close()
else:
    print("Data file found, load it!")
    file = open(data_folder + sep + "data.pkl", 'rb')
    X,y,texts = pickle.load(file)
    file.close()
# print("After loading raw data")
print(X.shape)
# print((X[10:20]))
# print((y[10:20]))
# print((texts[10:20]))
X_rest, X_val, y_rest, y_val = train_test_split(X, y, random_state=123, test_size=0.1,shuffle=True)

word_vectors = KeyedVectors.load_word2vec_format('../intentDetection/w2v/wiki.vi.model.bin', binary=True)

EMBEDDING_DIM=400
word_index = txtTokenizer(getData(file_path))[1]
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
print(vocabulary_size)
embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

del(word_vectors)

embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)


sequence_length = X.shape[1]
filter_sizes = [1,2,3,5]
num_filters = 64
drop = 0.5
def get_model():
    inputs = Input(shape=(sequence_length,))
    embedding = Embedding(vocabulary_size,EMBEDDING_DIM,weights=[embedding_matrix],trainable=False)(inputs)
    reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_3 = Conv2D(num_filters, (filter_sizes[3], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
    maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
    maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
    maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
    maxpool_3 = MaxPooling2D((sequence_length - filter_sizes[3] + 1, 1), strides=(1, 1))(conv_3)
    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2,maxpool_3], axis=1)
    flatten = Flatten()(merged_tensor)
    reshape = Reshape((4*num_filters,))(flatten)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=17, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)
    model = Model(inputs, output)

    adam = Adam(lr=1e-3)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['acc'])
    return model
# num_folds = 5
# acc_per_fold = []
# loss_per_fold = []
#
#
# kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
# fold_no = 1
#
# for train, test in kfold.split(X_rest, y_rest):
#
#     model = get_model()
#     callbacks = [EarlyStopping(monitor='val_loss')]
#     print('------------------------------------------------------------------------')
#     print(f'Training for fold {fold_no} ...')
#
#     # Fit data to model
#     model.fit(X_rest[train], y_rest[train], batch_size=32, epochs=10, verbose=10,callbacks=callbacks,validation_data=(X_val,y_val))
#     test_predictions_probas = model.predict(X_rest[test])
#     test_predictions = test_predictions_probas.argmax(axis=-1)
#
#     scores = model.evaluate(X_rest[test], y_rest[test], verbose=0)
#
#     print(
#         f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
#     acc_per_fold.append(scores[1] * 100)
#     loss_per_fold.append(scores[0])
#     y_test = y_rest[test].argmax(axis=-1)
#     # print(y_test)
#     # print("--------------")
#     # print(test_predictions)
#     target_names = intents
#     print("Precision, Recall and F1-Score:\n\n",classification_report(y_test, test_predictions, target_names=target_names))
#     fold_no = fold_no + 1
#
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#   print('------------------------------------------------------------------------')
#   print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('------------------------------------------------------------------------')
# #///////////////////////////
#     #
#     # in_channels = 1
#     # out_channels = 64
#     # # stride  =1
#     # # padding = 0
#     # word_embeddings = nn.Embedding(vocab_size, embedding_length)
#     # self.word_embeddings.weight = nn.Parameter(weights, requires_grad=False)
#     # conv1 = Conv2D(in_channels, out_channels, (filter_sizes[0], EMBEDDING_DIM), stride, padding)
#     # conv2 = Conv2D(in_channels, out_channels, (filter_sizes[1], EMBEDDING_DIM), stride, padding)
#     # conv3 = Conv2D(in_channels, out_channels, (filter_sizes[2], EMBEDDING_DIM), stride, padding)
#     # dropout = Dropout(0.5)
#     # self.label = Linear(4* out_channels, 17)
#     # ///////////////////////////
# model = get_model()
# model.save(data_folder + sep + "predict_model.save")
