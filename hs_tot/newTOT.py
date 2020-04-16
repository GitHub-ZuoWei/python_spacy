#!/usr/bin/env python
# coding: utf-8

import re
import json
import time
import math
import pymysql
import scipy.stats
import numpy as np
import scipy.special
from datetime import datetime
import matplotlib.pyplot as plt
from hs_tot.tot import TopicsOverTime
from utils.sql_helper import MySqlHelper
from hs_tot.every_day import getEveryDay
from matplotlib.font_manager import FontProperties


# sql_helper = MySqlHelper()
# sql_helper.connect_database()


def transtime(t):
    t = datetime.strptime(t, '%Y-%m-%d')
    # t = datetime.strptime(t, '%d/%m/%Y %H:%M:%S')
    t = datetime.strftime(t, '%Y-%m-%d')
    return t

# 可整合到TOT.py 中，代替GetPnasCorpusAndDictionary

def calculate_origin(data_list):
    # 遍历新闻，获取分词词语、时间戳
    dictionary = []
    timestamps = []
    documents = []
    data_list = sorted(data_list, key=lambda news: news['publicDateTime'])
    dictionary = set(dictionary)
    for news in data_list:
        ts = time.strptime(news['publicDateTime'], "%Y-%m-%d")
        timestamps.append(time.mktime(ts))
        words = [word for word in news['cutted_content']]
        documents.append(words)
        dictionary.update(words)
    first_timestamp = timestamps[0]
    last_timestamp = timestamps[len(timestamps) - 1]
    time_length = last_timestamp - first_timestamp
    timestamps = [1.0 * (t - first_timestamp + 0.05 * time_length) / (time_length * 1.1) for t in timestamps]
    dictionary = list(dictionary)
    origin_data = {'documents': documents, 'timestamps': timestamps,
                   'dictionary': dictionary}
    return origin_data


def get_keywords(phi, words, IDF, k=5):
    keywords = []
    for t_phi in phi:
        nums = [t_phi[i] * IDF[i] for i in range(len(words))]
        temp = []
        Inf = 0
        for i in range(k):
            temp.append(nums.index(max(nums)))
            nums[nums.index(max(nums))] = Inf
        key = [words[j] for j in temp]
        # print(key)
        keywords.append(key)
    return keywords


def VisualizeTopics(phi, words, num_topics, viz_threshold=9e-4):
    phi_viz = np.transpose(phi)
    words_to_display = ~np.all(phi_viz <= viz_threshold, axis=1)
    words_viz = [words[i] for i in range(len(words_to_display)) if words_to_display[i]]
    phi_viz = phi_viz[words_to_display]

    fig, ax = plt.subplots(figsize=(18, 30))
    #	fig = plt.figure(figsize=(18,15))
    heatmap = plt.pcolor(phi_viz, cmap=plt.cm.Blues, alpha=1)
    plt.colorbar()

    # fig.set_size_inches(8, 11)
    ax.grid(False)
    ax.set_frame_on(False)

    ax.set_xticks(np.arange(phi_viz.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(phi_viz.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    # plt.xticks(rotation=45)

    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    font = FontProperties(fname=r"hs_tot/Songti.ttc", size=14)
    column_labels = words_viz  # ['Word ' + str(i) for i in range(1,1000)]
    row_labels = ['Topic ' + str(i) for i in range(1, num_topics + 1)]
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False, fontproperties=font)
    # print(column_labels)
    # print(row_labels)

    plt.show()


def VisualizeEvolution(psi):
    xs = np.linspace(0, 1, num=1000)
    fig, ax = plt.subplots(figsize=(18, 12))
    #	fig = plt.figure(figsize=(18,15))
    for i in range(len(psi)):
        ys = [pow(x, (psi[i][0] - 1)) * pow(1 - x, (psi[i][1] - 1)) * math.sqrt(
            abs(psi[i][1] - psi[i][0])) / scipy.special.beta((psi[i][0]), (psi[i][1])) for x in xs]
        ax.plot(xs, ys, label='Topic ' + str(i + 1))

    ax.legend(loc='best', frameon=False)
    plt.show()


db = pymysql.connect('192.168.10.222', user='root', password='123456', db='usppa_dev')

cursor = db.cursor()


# 接收参数 计算结果 入库
def tot_result(start_time, end_time, person_id, receive_data):
    news_data = receive_data
    data = []
    for i in range(len(news_data)):
        one_news = {}
        if len(news_data[i]['content']) > 5 and re.findall('[\u4e00-\u9fa5]+', news_data[i]['content']) == []:
            one_news['id'] = i
            one_news['cutted_content'] = json.loads(news_data[i]['content'].lower())
            one_news['publicDateTime'] = transtime(news_data[i]['oriTime'])
            data.append(one_news)

    all_data = sorted(data, key=lambda news: news['publicDateTime'])

    all_data = [x for x in all_data if x['publicDateTime'] > '2018-05-01']
    tot_data = all_data

    origin_data = calculate_origin(tot_data)
    tot = TopicsOverTime()

    if len(news_data) >= 1000:
        num = 20  # 迭代次数
        topic_num = 15
    else:
        num = 20  # 迭代次数
        topic_num = 10

    # 初次训练
    par = tot.InitializeParameters(origin_data['documents'], origin_data['timestamps'], origin_data['dictionary'], num,
                                   topic_num)
    theta, phi, psi = tot.TopicsOverTimeGibbsSampling(par)

    # VisualizeTopics(par['phi'], par['word_token'], par['T'])
    # VisualizeEvolution(par['psi'])
    phi_t = np.transpose(par['phi'])
    dic_p = {}
    dictionary = origin_data['dictionary']
    for x in range(len(phi_t)):
        word = dictionary[x]
        pList = phi_t[x]
        dic_p[word] = pList
    topics = [[] for x in range(len(par['theta'][0]))]
    theta_data = par['theta']
    timestamps = origin_data['timestamps']
    doc_with_topic_mark = []
    for i in range(len(theta_data)):
        x = theta_data[i]
        index = list(x).index(max(x))
        item = tot_data[i]
        # 计算文章中的词，在每个专题中的概率
        content = item['cutted_content']
        item['timestamps'] = timestamps[index]
        item['word_num'] = len(content)
        ps = []
        for word in content:
            try:
                ps.append(dic_p[word.lower()])
            except:
                # print(word)
                continue
        nd = np.array(ps)
        p_list = np.sum(nd, axis=0)
        item['p_list'] = p_list
        item['p_max_topic'] = index
        item['p_max'] = max(x)
        topics[index].append(item)
        doc_with_topic_mark.append(index)

    topics_words_pro = []
    for j in range(topic_num):
        sum_words_num = 0
        sum_plist = np.asarray([0. for _ in range(topic_num)])
        for i in range(len(topics[j])):
            sum_plist += topics[j][i]['p_list']
            sum_words_num += topics[j][i]['word_num']
        topics_words_pro.append(list(sum_plist / sum_words_num))
    # 所有点
    # print(topics_words_pro)

    # plt.figure(figsize=(30, 10))
    # for topic in topics:
    #     t_topic = []
    #     topic_count = []
    #     x = []
    #     for news in topic:
    #         ts = time.strptime(news['publicDateTime'], "%Y-%m-%d")
    #         t_topic.append(time.mktime(ts) / 86400)
    #
    #     for i in set(t_topic):
    #         x.append(i)
    #         # print(i)
    #         topic_count.append(t_topic.count(i))
    #     plt.bar(x, topic_count, width=0.5, align="center", label='Topic ')

    # if topic:
    #     print(sorted(topic, key=lambda news: news['p_max'])[-1])
    # plt.show()

    # sdata = sorted(tot_data, key=lambda news: news['publicDateTime'])
    # # print(sdata)

    count_words = {word: 0 for word in par['word_token']}
    for news in tot_data:
        for word in set(news['cutted_content']):
            if word in count_words:
                count_words[word] += 1
    data_num = len(tot_data)
    # %%
    words_idf = {word: math.log(data_num / count_words[word]) for word in count_words}
    # %%
    IDF = [words_idf[word] for word in par['word_token']]
    # sdata = sorted(tot_data, key=lambda news: news['publicDateTime'])
    # for topic in topics:
    #     print(len(topic))
    #     if topic:
    #         print(sorted(topic, key=lambda news: news['p_max'])[-1])
    # one_topic = sorted(one_topic, key=lambda news: news['p_max'])
    # print(one_topic[-1])
    keywords = get_keywords(phi, par['word_token'], IDF, 5)

    tot_result_topic = {}
    tot_result_topic1 = {}
    # json_keywords=[]
    for i in range(len(topics)):
        if len(topics[i]) < 0.01 * data_num:
            # tot_result_topic['topic' + str(i + 1)] = {'value': [], 'keywords': []}
            # tot_result_topic1['topic' + str(i + 1)] = []
            continue
        else:
            # 每一个主题的关键字
            # json_keywords.append(keywords[i])
            N = []
            for news in topics[i]:
                N.append((news['publicDateTime'], keywords[i]))
            # print(N[-1][0])
            # tot_result_topic["keywords"] = keywords[i]

            key = i
            tot_result_topic1['topic' + str(key + 1)] = N

            every_date = []
            for every_N in range(len(N)):
                every_date.append(N[every_N][0])
            tot_result_topic['topic' + str(key + 1)] = {'value': every_date, 'keywords': keywords[i]}
    # 每一个主题的总数量
    # json_list = []
    # for i in tot_result_topic.items():
    #     json_list.append({"Topic" + str(i[0]): len(i[1])})

    # print(getEveryDay(start_time, end_time))

    time_list_num = {}
    # final_data['time']=x_date_list
    # print(x_date_list)
    for result_topic_data in tot_result_topic1.items():
        # print(result_topic_data)
        try:
            for num in range(1, 15):
                if result_topic_data[int(num)][1] != None:
                    for k in range(len(result_topic_data[int(num)])):
                        # 每一个主题下的所有数据 ('2018-05-09', ['deserve', 'smart', 'american', 'spectacular', 'andrews air force base'])
                        # print(result_topic_data[int(num)][k][0])
                        k_ = datetime.strptime(result_topic_data[int(num)][k][0], '%Y-%m-%d').date()
                        # print(k_, result_topic_data[int(num)].count("2018-05-15"))
                        # print( result_topic_data[int(num)][k])
                        # time_list_num['num'] = (result_topic_data[int(num)][k][0])
                        # print(result_topic_data[int(num)][k])

                    #     if num<15:
                    #         print(dateArr[num])
                    #         if k_>datetime.strptime(dateArr[num],'%Y-%m-%d').date() and k_<=datetime.strptime(dateArr[num+1],'%Y-%m-%d').date():
                    #             itemData[num] = itemData[num]+1
                    #             print(itemData[num])
                    # time_list_num["topic" + str(num)] = itemData

        except IndexError as e:
            pass
    # print(time_list_num)
    # print(json_keywords)
    # print(json.dumps(tot_result_topic))

    # 根据传入的开始和结束时间 分段
    x_date_list = getEveryDay(start_time, end_time)
    set_day_list = []
    for day in x_date_list:
        if day not in set_day_list:
            set_day_list.append(day)
    # final_data = {'startTime': start_time, 'endTime': end_time, 'topic': tot_result_topic, 'time': set_day_list}
    # print(final_data)
    ###############
    for topic, value in tot_result_topic.items():
        # print(topic)
        try:
            t = value['value']
            new_time = []
            for index, tt in enumerate(set_day_list):
                if index == len(set_day_list) - 1:
                    break
                num = 0
                for top_t in t:
                    if top_t < set_day_list[index + 1] and top_t >= tt:
                        num += 1
                new_time.append(num)
            value['value'] = new_time
        except Exception as e:
            pass

    final_data = {'startTime': start_time, 'endTime': end_time, 'topic': tot_result_topic, 'time': set_day_list[:14]}
    # print(final_data)
    #
    date_now = time.strftime('%Y-%m-%d %H:%M:%S')
    sql = "Insert Into usppa_person_tot (person_id,json,insert_time) VALUE ('%s', '%s', '%s')" % (
        person_id, json.dumps(final_data), date_now)

    # sql_helper.execute_sql(sql)
    # sql_helper.close()

    cursor.execute(sql)
    db.commit()
