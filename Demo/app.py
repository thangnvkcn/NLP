from flask import Flask, render_template, request,json,redirect
import pickle
from pre_processing_data import *
app = Flask(__name__)
with open('model_cnn.pkl', 'rb') as fp:
    cnn_17 = pickle.load(fp)
with open('model_cnn_45_intents.pkl', 'rb') as fp:
    cnn_45 = pickle.load(fp)
import time
start_time = time.time()
# with open('model_lstm.pkl_', 'rb') as fp:
#     lstm_17 = pickle.load(fp)
# with open('model_lstm_45_intents.pkl_', 'rb') as fp:
#     lstm_45 = pickle.load(fp)
end_time = time.time()
print('total run-time: %f ms' % ((end_time - start_time) * 1000))
# with open('model_CRF_NER.pkl', 'rb') as fp:
#     crf = pickle.load(fp)
import pymysql
mydb = pymysql.connect(
            host="sql12.freemysqlhosting.net",
            user="sql12363092",
            password="7NxMU3BuTu",
            database="sql12363092"
        )

# app.config['MYSQL_HOST'] = ''
# app.config['MYSQL_USER'] = ''
# app.config['MYSQL_PASSWORD'] =''
# app.config['MYSQL_DB'] = ''

@app.route('/')
def index():
    return render_template("index.html")


# @app.route('/process', methods=['GET','POST'])
# def process():
#     if request.method == 'POST':
#         choice = request.form['model_option']
#         rawtext = request.form['rawtext']
#         text = rawtext
#
#
#     return render_template("index.html", results_17=results_17,results_45=results_45,text = text )

@app.route('/process', methods=['GET','POST'])
def vote():
    if request.method == 'POST':

        rawtext=request.get_json()['text']
        choice = request.get_json()['model']
        print(choice)
        print(rawtext)
        if choice == 'cnn':
            test_sentence = [rawtext]
            results_17 = processing_cnn_17(test_sentence,cnn_17)
            results_45 = processing_cnn_45(test_sentence,cnn_45)
        elif choice == 'lstm':
            test_sentence = [rawtext]
            results_17 = processing_lstm_17(test_sentence)
            results_45 = processing_lstm_45(test_sentence)
        results_NER = processing_NER(rawtext)
        teacher_name = None
        campus_name = None
        dept_name = None
        for i in range(len(results_NER[1])):
            if (results_NER[1][i] == 'Major_Name'):
                major_name = results_NER[0][i].strip()
                major_name = major_name.replace("ngành ", "")
                major_name = major_name.replace("nganh ", "")

            if (results_NER[1][i] == 'datetime'):
                year = results_NER[0][i].strip()
                if (len(year) > 4):
                    year = year.split(' ', 1)[1]
            if (results_NER[1][i] == 'Teacher_Name'):
                teacher_name = results_NER[0][i].strip()
                print(teacher_name)
            if (results_NER[1][i] == 'Campus_Name' or results_NER[1][i] == 'Dept_Name'):
                campus_name = results_NER[0][i].strip()
                print(campus_name)
                campus_name = campus_name.replace("cs ", "")
                campus_name = campus_name.replace("cơ sở ", "")
            if (results_NER[1][i] == 'Dept_Name'):
                dept_name = results_NER[0][i].strip()

        if results_17 == "Basic_point":
            answer = read_Basic_point(mydb,major_name,year)
        if results_17 == "Major_infor":
            answer = read_Major_Infor(mydb, major_name)
        if results_17 == "contact_point" and campus_name is not None and teacher_name is None:
            answer = read_Contact_point_Dept(mydb, campus_name)
        if results_17 == "contact_point" and teacher_name is not None:
            answer = read_Contact_point_Teacher(mydb, teacher_name, dept_name)



        st1 = results_NER[0]
        st2 = results_NER[1]
        text=""
        p='<p align="center" class="tag_{}">{} <br> {}</p>'

        for i in range(len(st2)):
            label=st2[i]
            word=st1[i]
            if label=="O":
                text += p.format(label, word, "")
            else:
                text+=p.format(label,word,label)
        text={'text':text,'results_17':results_17,'results_45':results_45,'answer':answer}
    return json.dumps({'success': True, 'text_tagged': text}), 200, {'Content-Type': 'application/json; charset=UTF-8'}
if __name__ == '__main__':
    app.run(threaded=False)