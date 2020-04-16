# -*- coding:utf-8 -*-
# Author      : suwei<suwei@yuchen.net.cn>
# Datetime    : 2020-04-13 14:51
# User        : suwei
# Product     : PyCharm
# Project     : LtpBertServer_v1
# File        : 1
# Description : ltp
import os

from pyltp import Segmentor, Postagger, NamedEntityRecognizer


class LtpUtil:
    def __init__(self, stopwords_file=None, lexicon_pos_file=None):
        '''

        :param ltp_module_path: LTP模型文件路径
        :param stopwords_file: 停用词列表
        :param lexicon_pos_file: 自定义实体字典
        '''
        ltp_module_path = '/export/python/ltp_data_v3.4.0/'
        cws_model_path = os.path.join(ltp_module_path, 'cws.model')  # 分词
        user_dict_path = os.path.join(ltp_module_path, 'lexicon')  # 分词
        self.segmentor = Segmentor()
        # self.segmentor.load(cws_model_path)
        self.segmentor.load_with_lexicon(cws_model_path, user_dict_path)

        self.postagger = Postagger()
        self.postagger.load(os.path.join(ltp_module_path, "pos.model"))  # 词性
        # self.postagger.load_with_lexicon(os.path.join(ltp_module_path, "pos.model"),
        #                                  os.path.join(ltp_module_path, "lexicon_bert"))
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(ltp_module_path, "ner.model"))  # 实体识别

        if stopwords_file:
            self.stopwords = [line.strip() for line in open(stopwords_file, 'r', encoding='utf-8').readlines()]
        else:
            self.stopwords = None

    def get_words(self, sentence):
        words = self.segmentor.segment(sentence)
        if self.stopwords:
            w_list = list(
                filter(lambda x: x not in self.stopwords
                                 and x != '\t' and x != '\n' and x != '\xa0', words))
        else:
            w_list = list(words)
        return w_list

    def get_postags(self, words):
        post = self.postagger.postag(words)
        return list(post)

    def get_ners(self, words, postags):
        netags = self.recognizer.recognize(words, postags)
        return list(netags)


if __name__ == '__main__':
    sentence = '金庸的作品里基本每一名角色的名字都有独特用意，或引经据典、或五行相配、或诗情画意、或言简意赅。综合来说，都较合人物身份与经历。'
    ltp = LtpUtil()
    words = ltp.get_words(sentence)
    print(words)
    postags = ltp.get_postags(words)
    print(postags)
    ners = ltp.get_ners(words, postags)
    print(ners)

    for a, b, c in zip(words, postags, ners):
        print(a, '\t\t', b, '\t\t', c)
