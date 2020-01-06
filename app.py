import datetime
import json
import os
import shutil
import time
import spacy
from gevent import pywsgi
from flask import Flask, request
from hs_tot.newTOT import tot_result
from util_textrank import TextRankMain
from base_app import return_web
from concurrent.futures import ThreadPoolExecutor
from utils.sql_helper import MySqlHelper

sql_helper = MySqlHelper()
app = Flask(__name__)
nlp = spacy.load('en_core_web_sm')  # 加载预训练模型

executor = ThreadPoolExecutor(max_workers=66)  # 创建线程池


def get_ners_pool(content):
    doc = nlp(content.strip())
    # 实体识别
    ners = [{"entity": ent.text, "type": ent.label_} for ent in doc.ents]
    print(os.getpid())
    # time.sleep(10)
    return return_web(ners, 20000)


# 实体识别
@app.route('/dev/getentity', methods=['POST'])
def get_ners():
    if request.method == 'POST':
        try:
            data = request.get_data()
            json_data = json.loads(data.decode('utf-8'))
            content = json_data.get("content")
            language = json_data.get("language")
        except Exception:
            return return_web("未知异常错误", code=0)
        if language == "EN":
            doc = nlp(content.strip().replace('<p>', ' ').replace('</p>', ''))
            # 实体识别
            ners = [{"entity": ent.text, "type": ent.label_} for ent in doc.ents]
            return return_web(ners, 20000)
            # submit = executor.submit(get_ners_pool, content)
            # return submit.result()
        if language == "CHS":
            ners = [{"entity": "中文呦", "type": "中文呦"}]
            return return_web(ners, 20000)
        else:
            msg = [{"msg": "参数输入错误"}]
            return return_web(msg, 40400)

    return return_web("请求方式错误", code=40000)


# 词性标注
@app.route('/dev/getpostagging', methods=['POST'])
def get_postagging():
    if request.method == 'POST':
        try:
            data = request.get_data()
            json_data = json.loads(data.decode('utf-8'))
            content = json_data.get("content")
            language = json_data.get("language")
        except Exception:
            return return_web("未知异常错误", code=0)
        if language == "EN":
            doc = nlp(content.strip().replace('<p>', ' ').replace('</p>', ''))
            # 词性标注
            pos = [{"postagging": token.pos_, "original": token.lemma_} for token in doc]
            return return_web(pos, 20000)
        if language == "CHS":
            ners = [{"postagging": "中文呦", "original": "中文呦"}]
            return return_web(ners, 20000)
        else:
            msg = [{"msg": "参数输入错误"}]
            return return_web(msg, 40400)

    return return_web("请求方式错误", code=40000)


# 分词
@app.route('/dev/getparticiple', methods=['POST'])
def get_getparticiple():
    if request.method == 'POST':
        try:
            data = request.get_data()
            json_data = json.loads(data.decode('utf-8'))
            content = json_data.get("content")
            language = json_data.get("language")
        except Exception:
            return return_web("未知异常错误", code=0)
        if language == "EN":
            doc = nlp(content.strip())
            # 词性标注
            tokens = [{"content": [str(token) for token in doc]}]
            return return_web(tokens, 20000)
        if language == "CHS":
            tokens = [{"content": "中文呦"}]
            return return_web(tokens, 20000)
        else:
            msg = [{"msg": "参数输入错误"}]
            return return_web(msg, 40400)

    return return_web("请求方式错误", code=40000)


# 提取关键句
@app.route('/dev/getsentence', methods=['POST'])
def get_getsentence():
    if request.method == 'POST':
        try:
            data = request.get_data()
            json_data = json.loads(data.decode('utf-8'))
            content = json_data.get("content")
            language = json_data.get("language")
        except Exception:
            return return_web("未知异常错误", code=0)
        if language == "EN":
            doc = nlp(content.strip())
            sents = []
            for sent in doc.sents:
                sents.append(sent.text.strip())
            sents = '\n'.join(sents)
            result = TextRankMain().key_sentences(sents, 3)
            if result:
                return return_web(result[0], 20000)
            else:
                return return_web('没有提取到关键句', 20000)
        if language == "CHS":
            msg = [{"content": "中文呦"}]
            return return_web(msg, 20000)
        else:
            msg = [{"msg": "参数输入错误"}]
            return return_web(msg, 40400)

    return return_web("请求方式错误", code=40000)

# TOT
@app.route('/dev/gettot', methods=['POST'])
def get_gettot():
    if request.method == 'POST':
        try:
            data = request.get_data()
            json_data = json.loads(data.decode('utf-8'))
            start_time = json_data.get("startTime")
            end_time = json_data.get("endTime")
            person_id = json_data.get("personId")
            receive_data = json_data.get("data")
        except Exception:
            return return_web("未知异常错误", code=0)
        if receive_data:
            executor.submit(get_tot_result, start_time, end_time, person_id, receive_data)
            msg = [{"MSG": "success"}]
            return return_web(msg, 20000)

    return return_web("请求方式错误", code=40000)


# TOT异步调用
def get_tot_result(start_time, end_time, person_id, receive_data):
    tot_result(start_time, end_time, person_id, receive_data)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 666), app)
    server.serve_forever()
