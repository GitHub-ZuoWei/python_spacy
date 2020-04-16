'''
TextRank算法是一种文本排序算法，
由谷歌的网页重要性排序算法PageRank算法改进而来，
它能够从一个给定的文本中提取出该文本的关键词、关键词组，
并使用抽取式的自动文摘方法提取出该文本的关键句
'''
from textrank4zh import TextRank4Keyword, TextRank4Sentence


class TextRankMain:
    def __init__(self, word_num=10, sen_num=10):
        # 加入日志

        # 初始化
        self.word_num = word_num
        self.sen_num = sen_num
        self.tr4w = TextRank4Keyword()
        self.tr4s = TextRank4Sentence()

    # 获取文本中的关键词和关键短语
    def get_keywords_and_phrases(self, text, size, only_word=True):
        """
        获取文本中的关键词和关键短语
        :param text:
        :return:
        """
        words_phrases = {}
        words = []
        phrases = []

        self.tr4w.analyze(text, lower=True, window=2)
        # 从关键词列表中获取前20个关键词
        for item in self.tr4w.get_keywords(num=size, word_min_len=1):
            # 打印每个关键词的内容及关键词的权重
            # print(item.word, item.weight)
            words.append({'word': item.word})

        words_phrases['words'] = words

        if only_word:
            return words_phrases
        else:
            # 从关键短语列表中获取关键短语
            for phrase in self.tr4w.get_keyphrases(keywords_num=self.word_num, min_occur_num=1):
                # print(phrase)
                phrases.append(phrase)

            words_phrases['phrases'] = phrases
            return words_phrases

    def key_sentences(self, text, source='all_filters'):
        """
        获取文本中的关键句子
        :param text: 文本
        :param save_num: 获取的条数
        :param source:默认值为`'all_filters'`:剔除停用词并进行词性过滤，可选值为`'no_filter', 'no_stop_words', 'all_filters'`。
        :return:
        """
        # lower=True:英文单词小写
        self.tr4s.analyze(text=text, lower=True, source=source)
        sentences = []
        # 抽取3条句子作为摘要
        for item in self.tr4s.get_key_sentences(num=self.sen_num):
            # 打印句子的索引、权重和内容
            sentences.append(item)
        return sentences


'''
if __name__ == '__main__':
    # data_list = text_dbserver.text_get_source(1, 10000)
    o = TextRankMain()
    # text = '这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。答谢宴于晚上8点开始。'
    text = '习近平指出，永葆政治本色,永葆政治本色,永葆政治本色,政治上的坚定、党性上的坚定都离不开理论上的坚定。' \
           '忠诚和信仰是具体的、实践的。要经常对照党章党规党纪，中国共产党人的理想信念建立在对马克思主义的深刻理解之上检视自己的理想信念和思想言行，' \
           '不断掸去思想上的灰尘，永葆政治本色。这间酒店位于北京东三环，里面摆放很多雕塑，文艺气息十足。答谢宴于晚上8点开始。'

    data_list = [{'content': '阿斯顿发的 打发打发打发第三方大发送到'},
                 {"content": '阿斯顿发的 打发打发打发第三方大发送到'}]
    # print(o.get_keywords_and_phrases(text))
    b = o.key_sentences(text, 3)
'''
