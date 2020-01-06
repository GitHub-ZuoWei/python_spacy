# -*- coding:utf-8 -*-
# Author      : suwei<suwei@yuchen.net.cn>
# Datetime    : 2019-07-18 14:36
# User        : suwei
# Product     : PyCharm
# Project     : Demo_BI
# File        : base_app.py
# Description : app.py 的一些辅助方法
import json

from flask import Response


def return_web(data, code=0):
    content = {'code': code, 'data': data}
    return Response(json.dumps(content, ensure_ascii=False), content_type='application/json')
