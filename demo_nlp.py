# -*- coding:utf-8 -*-
# Author      : suwei<suwei@yuchen.net.cn>
# Datetime    : 2019-10-17 16:07
# User        : suwei
# Product     : PyCharm
# Project     : Demo_spacy
# File        : demo_spacy.py
# Description : 英文实体识别-spacy


import spacy

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型

s = '''
"If you want to exist in China as a global distributor, you must always be at the Canton Fair," said Mr. Dariusz Hajek, Vice President and Purchase Executive of TOYA, who has been attending the Canton Fair for over twenty years now.

TOYA group, a Poland-based company, is one of the world's leading manufacturers of tools and power tools. This year they will exhibit their line of YATO branded manual and electric tools such as wrenches, screwdrivers and electric glue guns at the fair.
'''

# s = 'I love shuxue $. and love math.'

doc = nlp(s.strip())
print('-------------分句-------------')
for sent in doc.sents:
    print(sent.text.strip())
    print('-*-' * 10)

print('-------------分词-------------')
tokens = [token for token in doc]
print(tokens)

print('-------------词性标注-------------')
pos = [token.pos_ for token in doc]
print(pos)

print('-------------实体识别-------------')
ners = [(ent.text, ent.label_) for ent in doc.ents]  # ents
print(ners)
