# -*- coding:utf-8 -*-
# Author      : suwei<suwei@yuchen.net.cn>
# Datetime    : 2019-07-18 13:40
# User        : suwei
# Product     : PyCharm
# Project     : Demo_BI
# File        : sql_helper.py
# Description : 数据库连接工具类
import pymysql

import config


class MySqlHelper:

    def __init__(self):
        # 连接数据库
        self.conn = None
        self.cursor = None
        pass

    def connect_database(self):
        try:
            self.conn = pymysql.connect(config.sql_info['ip'], config.sql_info['user'],
                                        config.sql_info['pwd'], config.sql_info['database'])
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            print('数据库连接失败：', e)
            return False

    def execute_sql(self, sql):
        try:
            self.conn.ping(reconnect=True)
            self.cursor.execute(sql)
            return self.cursor.fetchall()
        except Exception as e:
            self.conn.rollback()
            print('SQL语句执行失败：', e, '\n\t\t', sql)

    def close(self):
        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()
