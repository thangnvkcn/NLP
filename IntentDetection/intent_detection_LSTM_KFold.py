from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense,Embedding,Bidirectional,Dropout
from gensim.models.keyedvectors import KeyedVectors
from keras.utils import to_categorical
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
# print(X.shape)
# print((X[10:20]))
# print((y[10:20]))
# print((texts[10:20]))
X_rest, X_val, y_rest, y_val = train_test_split(X, y, random_state=123, test_size=0.1,shuffle=True)

word_vectors = KeyedVectors.load_word2vec_format('../intentDetection/w2v/wiki.vi.model.bin', binary=True)

EMBEDDING_DIM=400
word_index = txtTokenizer(getData(file_path))[1]
vocabulary_size=min(len(word_index)+1,NUM_WORDS)
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

def get_model():
    model = Sequential()
    model.add(Embedding(vocabulary_size,EMBEDDING_DIM,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
    model.add(Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(17, activation='softmax'))
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

num_folds = 5
acc_per_fold = []
loss_per_fold = []

seed = 123

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
fold_no = 1
best_model = ""
best_score = 0
for train, test in kfold.split(X_rest, y_rest):
      model = get_model()
      model.fit(X_rest[train], y_rest[train], epochs=10, batch_size=32, verbose=0, validation_data=(X_val, y_val))
     # evaluate the model
      test_predictions_probas = model.predict(X_rest[test])
      test_predictions = test_predictions_probas.argmax(axis=-1)
     # Generate generalization metrics
      scores = model.evaluate(X_rest[test], y_rest[test], verbose=0)
      print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
      acc_per_fold.append(scores[1] * 100)
      loss_per_fold.append(scores[0])
      y_test = y_rest[test].argmax(axis=-1)
      # print(y_test)
      # print("--------------")
      # print(test_predictions)
      target_names = intents
      print("Precision, Recall and F1-Score:\n\n",classification_report(y_test, test_predictions, target_names=target_names))
      # Increase fold number
      fold_no = fold_no + 1


print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
