from keras.preprocessing.sequence import pad_sequences
from underthesea import pos_tag
from underthesea import word_tokenize
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import pickle
NUM_WORDS = 1500
def getData(file_path):
    data = pd.read_csv(file_path)
    return data
dic={}

def getLabels(data):
    file_path = 'data_intent_17.csv'
    d = getData(file_path)
    intents = d.intent.unique()
    texts = data.sentence
    for i,intent in enumerate(intents):
        dic[intent] = i
    labels = data.intent.apply(lambda x:dic[x])
    return texts,labels
def txtTokenizer(data):
    sentences = data.sentence
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    return tokenizer,word_index
def processing_cnn_17(test_sentence,model):
    file_path = 'data_intent_17.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256, dtype="long", truncating="post", padding="post")
    import time
    start_time = time.time()
    # with open('model_cnn.pkl', 'rb') as fp:
    #     model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent
def processing_cnn_45(test_sentence,model):
    file_path = 'data_intent_45.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256, dtype="long", truncating="post", padding="post")
    import time
    start_time = time.time()

    # with open('model_cnn_45_intents.pkl', 'rb') as fp:
    #     model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent
def processing_lstm_17(test_sentence):
    file_path = 'data_intent_17.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256)
    import time
    start_time = time.time()
    with open('model_lstm.pkl_', 'rb') as fp:
        model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent
def processing_lstm_45(test_sentence):
    file_path = 'data_intent_45.csv'
    data = getData(file_path)
    intents = data.intent.unique()
    tokenizer, word_index = txtTokenizer(data)
    X_test = tokenizer.texts_to_sequences(test_sentence)
    X_test = pad_sequences(X_test, maxlen=256)
    import time
    start_time = time.time()
    with open('model_lstm_45_intents.pkl_', 'rb') as fp:
        model = pickle.load(fp)
    predict = model.predict(X_test)
    predict = predict.argmax(axis=-1)
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    for indx, label in enumerate(intents):
        if (predict == indx):
            intent = label
    return intent


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    for x in unique_list:
        return x

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def processing_NER(sentence):

    pos = pos_tag(sentence)
    sentence = word_tokenize(sentence)
    X = [sent2features(pos)]
    import time
    start_time = time.time()
    with open('model_CRF_NER.pkl', 'rb') as fp:
        crf = pickle.load(fp)
    pred = crf.predict(X)
    pred = np.array(pred)
    pred = pred.flatten()
    end_time = time.time()
    print('total run-time: %f ms' % ((end_time - start_time) * 1000))
    # Product perform by upper word
    sentence2string = ''
    words = []
    tag = []
    i = 0
    if(len(pred)>=2):
        for word, label in list(zip(sentence, pred)):
            if label[0] == 'B':
                sentence2string = ''
                sentence2string += (word)
                tag.append(label[2:])
            if label[0] == 'I' and word!=',':
                sentence2string += (' '+word)
            if label[0] == 'I' and word==',':
                sentence2string += (word)
            if label[0] == 'I' and (i+1 == len(pred)):
                words.append(sentence2string)
            if((i+1)>len(pred)):
                break
            if ((i + 1) < len(pred)):
                if label[0] == 'I' and pred[i + 1][0] == 'O':
                    words.append(sentence2string)
                if label[0] == 'I' and pred[i + 1][0] == 'B':
                    words.append(sentence2string)
                if label[0] == 'B' and pred[i + 1][0] == 'O':
                    words.append(sentence2string)
                if label[0] == 'B' and pred[i + 1][0] == 'B':
                    words.append(sentence2string)
            if ((i + 1) == len(pred)):
                if label[0] == 'B':
                    words.append(sentence2string)
            i = i + 1
        return words, tag
    if (len(pred) < 2):
        for word, label in list(zip(sentence, pred)):
            if label[0] == 'B':
                tag = []
                sentence2string = ''
                sentence2string += (word + ' ')
                tag.append(label[2:])
                words.append(sentence2string)
        return words, tag



def read_Basic_point(conn,tennganh,nam):
    print(tennganh)
    print("Read")
    cursor = conn.cursor()
    cursor.execute(f"select DiemChuan,TenNganh,Nam from Basic_point where TenNganh LIKE N'%{tennganh}%' AND Nam LIKE {nam};")
    for row in cursor:
        answer = f'Điểm chuẩn ngành {row[1]} năm {row[2]} là {row[0]}'
    return answer
def read_Major_Infor(conn,tennganh):
    print("Read")
    cursor = conn.cursor()
    cursor.execute(f"select MaNganh,TenNganh from Major_infor where TenNganh LIKE N'%{tennganh}%';")
    for row in cursor:
        answer = f'Mã ngành {row[1]} là {row[0]}'
    return answer
def read_Contact_point_Dept(conn,campus_name):
    print("Read")
    cursor = conn.cursor()
    cursor.execute(f"select Phong,SDT from Contact_point_Dept where Phong LIKE N'%{campus_name}%';")
    for row in cursor:
        answer = f'Số điện thoại {row[0]} là {row[1]}'
    return answer
def read_Contact_point_Teacher(conn,teacher_name,dept_name):
    print("Read")
    teacher_name_tmp = teacher_name.replace("thầy ", "")
    teacher_name_tmp = teacher_name_tmp.replace("cô ", "")
    teacher_name_tmp = teacher_name_tmp.replace("thây ", "")
    teacher_name_tmp = teacher_name_tmp.replace("thày ", "")
    teacher_name_tmp = teacher_name_tmp.replace("co ", "")
    cursor = conn.cursor()
    if(dept_name is None):
        cursor.execute(f"select SDT,Email from Contact_point_Teacher where HoVaTen LIKE N'%{teacher_name_tmp}%';")
        for row in cursor:
            answer = f'Số điện thoại của {teacher_name} là {row[0]} và email là {row[1]}'
    else:
        cursor.execute(f"select SDT,Email,BoPhan from Contact_point_Teacher where HoVaTen LIKE N'%{teacher_name_tmp}%' AND BoPhan LIKE N'%{dept_name}%';")
        for row in cursor:
            answer = f'Số điện thoại của {teacher_name} ở {row[2]} là {row[0]} và email là {row[1]}'
    return answer

# if __name__ == '__main__':
#     d ="Số điện thoại thầy Tuấn ở Phòng Tổ chức - Hành chính?"
#     d1 = [d]
#     a = processing_cnn_17(d1)
#     b  = processing_NER(d)
#     print(a)
#     print(b)
#     teacher_name  = None
#     campus_name  = None
#     dept_name = None
#     for i in range(len(b[1])):
#         if(b[1][i]=='Major_Name'):
#             major_name = b[0][i].strip()
#             major_name = major_name.replace("ngành ","")
#             major_name = major_name.replace("nganh ", "")
#
#         if (b[1][i] == 'datetime'):
#             year = b[0][i].strip()
#             if(len(year)>4):
#                 year = year.split(' ',1)[1]
#         if (b[1][i] == 'Teacher_Name'):
#             teacher_name = b[0][i].strip()
#             print(teacher_name)
#         if (b[1][i] == 'Campus_Name' or b[1][i] == 'Dept_Name'):
#             campus_name = b[0][i].strip()
#             print(campus_name)
#             campus_name = campus_name.replace("cs ", "")
#             campus_name = campus_name.replace("cơ sở ", "")
#         if (b[1][i] == 'Dept_Name'):
#             dept_name = b[0][i].strip()
#     import pyodbc
#     server = 'THANG-NGUYEN'
#     database = 'ChatBotDB'
#     username = 'sa'
#     password = '123'
#     driver = '{SQL Server Native Client 11.0}'  # Driver you need to connect to the database
#     port = '1433'
#     cnn = pyodbc.connect(
#         'DRIVER=' + driver + ';PORT=port;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username +
#         ';PWD=' + password)
#     if a=="Basic_point":
#         print(read_Basic_point(cnn,major_name,year))
#     if a=="Major_infor":
#         print(read_Major_Infor(cnn,major_name))
#     if a=="contact_point" and campus_name is not None and teacher_name is None:
#         print(read_Contact_point_Dept(cnn,campus_name))
#     if a=="contact_point" and teacher_name is not None:
#         print(read_Contact_point_Teacher(cnn,teacher_name,dept_name))


