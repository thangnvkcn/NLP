import gensim, re
import numpy as np
import pandas as pd
import  pickle
from  os import listdir

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import os
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense,Embedding

sep  = os.sep
data_folder = "data"

def txtTokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index

def preProcess(sentences):
    text = [re.sub(r'([^\s\w]|_)+', '', sentence) for sentence in sentences if sentence!='']
    text = [sentence.lower().strip().split() for sentence in text]
    return text

def loadData(data_folder):
    texts = []
    labels = []
    for folder in listdir(data_folder):
        if folder != ".DS_Store":
            print("Load cat: ",folder)
            for file in listdir(data_folder+sep+folder):
                if file != ".DS_Store":
                    print("Load file: ",file)
                    with open(data_folder + sep + folder + sep + file,'r',encoding= "utf-8") as f:
                        all_of_it = f.read()
                        sentences = all_of_it.split('.')
                        sentences = preProcess(sentences)
                        texts = texts + sentences
                        label = [folder for _ in sentences]
                        labels =  labels + label
                        del all_of_it,sentences
    return texts,labels

if not os.path.exists(data_folder + sep +"data.pkl"):
    print("Data file not found, build it!")
    texts, labels = loadData(data_folder)
    tokenizer, word_index = txtTokenizer(texts)
    X = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(X)

    y = pd.get_dummies(labels)
    file  =  open(data_folder + sep + "data.pkl", 'wb')
    pickle.dump([X,y,texts],file)
    file.close()
else:
    print("Data file found, load it!")
    file = open(data_folder + sep + "data.pkl", 'rb')
    X,y,texts = pickle.load(file)
    file.close()


print("After loading raw data")
print(X.shape)
print((X[10:20]))
print((y[10:20]))
print((texts[10:20]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle = True)

if not os.path.exists(data_folder + sep +"word_model.save"):
    word_model = gensim.models.Word2Vec(texts,size=300,min_count=1,iter=10)
    word_model.save(data_folder + sep + "word_model.save")
else:
    word_model = gensim.models.Word2Vec.load(data_folder + sep + "word_model.save")

print(word_model.wv.most_similar('con'))
# sys.exit()

embedding_matrix = np.zeros((len(word_model.wv.vocab)+1,300))
for i, vec in enumerate(word_model.wv.vectors):
    embedding_matrix[i] = vec
if not os.path.exists(data_folder + sep + "predict_model.save"):
    model = Sequential()
    model.add(Embedding(len(word_model.wv.vocab)+1,300,input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
    model.add(LSTM(300, return_sequences=False))
    model.add(Dense(y.shape[1],activation="softmax"))
    model.summary()
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['acc'])

    batch = 64
    epochs = 1
    model.fit(X_train,y_train,batch,epochs)
    model.save(data_folder + sep + "predict_model.save")
else:
    model = load_model(data_folder + sep + "predict_model.save")
# result = model.evaluate(X_test,y_test)
# print(result)