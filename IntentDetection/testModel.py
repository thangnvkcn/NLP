import os
import  numpy as np
import pandas as pd
from keras.models import load_model
from IntentDetection.intent_detection_usingKFold import get_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
sep  = os.sep
data_folder = "Data"
sep  = os.sep
data_folder = "Data"
file_path = '../intentDetection/Data/data_intent_17.csv'
NUM_WORDS = 1500
def getData(file_path):
    data = pd.read_csv(file_path)
    return data
# if not os.path.exists(data_folder + sep + "predict_model.save"):
#     model = get_model()
#     model.save(data_folder + sep + "predict_model.save")
# else:
model = load_model(data_folder + sep + "predict_model.save")

print("Nhập câu: ")
sentence = input()

X = pad_sequences(X)
print(X.shape)
# test_predictions_probas = model.predict(X)
# test_predictions = test_predictions_probas.argmax(axis=-1)
# print(test_predictions)