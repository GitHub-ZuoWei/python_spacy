# Copyright 2015 Abhinav Maurya

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import fileinput
import random
import scipy.special
import math
import numpy as np
import scipy.stats
import pickle
from math import log
import copy
from copy import deepcopy


class TopicsOverTime:
    def GetPnasCorpusAndDictionary(self, documents_path, timestamps_path, stopwords_path):
        documents = []
        timestamps = []
        dictionary = set()
        stopwords = set()
        for line in fileinput.FileInput(stopwords_path):
            stopwords.update(set(line.lower().strip().split()))
        for doc in fileinput.FileInput(documents_path):
            words = [word for word in doc.strip().split()]
            documents.append(words)
            dictionary.update(set(words))
        for timestamp in fileinput.FileInput(timestamps_path):
            num_titles = int(timestamp.strip().split()[0])
            timestamp = float(timestamp.strip().split()[1])
            timestamps.extend([timestamp for title in range(num_titles)])
        for line in fileinput.FileInput(stopwords_path):
            stopwords.update(set(line.lower().strip().split()))
        first_timestamp = timestamps[0]
        last_timestamp = timestamps[len(timestamps) - 1]
        timestamps = [1.0 * (t - first_timestamp) / (last_timestamp - first_timestamp) for t in timestamps]
        dictionary = list(dictionary)
        assert len(documents) == len(timestamps)
        return documents, timestamps, dictionary

    def CalculateCounts(self, par):
        for d in range(par['D']):
            for i in range(par['N'][d]):
                topic_di = par['z'][d][i]  # topic in doc d at position i
                word_di = par['w'][d][i]  # word ID in doc d at position i
                par['m'][d][topic_di] += 1
                par['n'][topic_di][word_di] += 1
                par['n_sum'][topic_di] += 1

    def InitializeParameters(self, documents, timestamps, dictionary,
                             max_number=25, topic_num=16, entity_words_id=[], entity_rank = 1.5):
        par = {}  # dictionary of all parameters
        par['dataset'] = 'pnas'  # dataset name
        par['max_iterations'] = max_number  # max number of iterations in gibbs sampling
        par['T'] = topic_num  # number of topics
        par['D'] = len(documents)  # documents:分词的列表
        par['V'] = len(dictionary)  # dictionary:词的不重复集合
        par['N'] = [len(doc) for doc in documents]
        par['alpha'] = [30.0 / par['T'] for _ in range(par['T'])]
        par['beta'] = [1.0 / par['T'] for _ in range(par['V'])]
        par['beta_sum'] = sum(par['beta'])
        par['psi'] = [[1 for _ in range(2)] for _ in range(par['T'])]
        par['betafunc_psi'] = [scipy.special.beta(par['psi'][t][0], par['psi'][t][1]) for t in range(par['T'])]
        par['word_id'] = {dictionary[i]: i for i in range(len(dictionary))}
        par['word_token'] = dictionary
        par['z'] = [[random.randrange(0, par['T']) for _ in range(par['N'][d])] for d in range(par['D'])]
        par['t'] = [[timestamps[d] for _ in range(par['N'][d])] for d in range(par['D'])]
        par['w'] = [[par['word_id'][documents[d][i]] for i in range(par['N'][d])] for d in range(par['D'])]
        par['m'] = [[0 for t in range(par['T'])] for d in range(par['D'])] # 为计算theta 计数
        par['n'] = [[0 for v in range(par['V'])] for t in range(par['T'])] # 为计算phi 计数
        par['theta'] = [[0 for t in range(par['T'])] for d in range(par['D'])]
        par['phi'] = [[0 for v in range(par['V'])] for t in range(par['T'])]
        par['n_sum'] = [0 for t in range(par['T'])]
        par['rank'] = [[1 for v in range(par['V'])] for t in range(par['T'])]
        par['entity_id'] = [0 for _ in range(par['V'])]
        par['doc_mark'] = [0 for _ in range(par['D'])]
        par['topic_mark'] = []
        np.set_printoptions(threshold=np.inf)
        np.seterr(divide='ignore', invalid='ignore')
        self.CalculateCounts(par)
        self.SetEntityRank(par, entity_words_id, entity_rank)
        return par

    def GetTopicTimestamps(self, par):
        topic_timestamps = []
        for topic in range(par['T']):
            current_topic_timestamps = []
            current_topic_doc_timestamps = [[(par['z'][d][i] == topic) * par['t'][d][i] for i in range(par['N'][d])] for
                                            d in range(par['D'])]
            for d in range(par['D']):
                current_topic_doc_timestamps[d] = list(filter(lambda x: x != 0, current_topic_doc_timestamps[d]))
            for timestamps in current_topic_doc_timestamps:
                current_topic_timestamps.extend(timestamps)
            if current_topic_timestamps == []:
                current_topic_timestamps = [random.random()*0.8+0.05]
                # print("Too many topics!")
                # print(topic)
            #assert current_topic_timestamps != []
            topic_timestamps.append(current_topic_timestamps)
        # print(len(topic_timestamps[1]))
        # print(len(topic_timestamps[0]))
        return topic_timestamps

    def GetMethodOfMomentsEstimatesForPsi(self, par):
        topic_timestamps = self.GetTopicTimestamps(par)
        psi = [[1 for _ in range(2)] for _ in range(len(topic_timestamps))]
        for i in range(len(topic_timestamps)):
            current_topic_timestamps = topic_timestamps[i]
            timestamp_mean = np.mean(current_topic_timestamps)
            timestamp_var = np.var(current_topic_timestamps)
            if timestamp_var == 0:
                timestamp_var = 1e-9
            if timestamp_mean == 0:
                timestamp_mean = 1e-9
            if timestamp_mean == 1:
                timestamp_mean = 1 - 1e-9
            common_factor = timestamp_mean * (1 - timestamp_mean) / timestamp_var - 1
            if common_factor == 0:
                common_factor == 1e-9
            psi[i][0] = 0.5 + abs(timestamp_mean * common_factor)
            psi[i][1] = 0.5 + abs((1 - timestamp_mean) * common_factor)

        return psi

    def ComputePosteriorEstimatesOfThetaAndPhi(self, par):
        theta = deepcopy(par['m'])
        phi = deepcopy(par['n'])

        for d in range(par['D']):
            if sum(theta[d]) == 0:
                theta[d] = np.asarray([1.0 / len(theta[d]) for _ in range(len(theta[d]))])
            else:
                theta[d] = np.asarray(theta[d])
                theta[d] = 1.0 * theta[d] / sum(theta[d])
        theta = np.asarray(theta)

        for t in range(par['T']):
            if sum(phi[t]) == 0:
                phi[t] = np.asarray([1.0 / len(phi[t]) for _ in range(len(phi[t]))])
            else:
                phi[t] = np.asarray(phi[t])
                phi[t] = 1.0 * phi[t] / sum(phi[t])
        phi = np.asarray(phi)

        return theta, phi

    def ComputePosteriorEstimatesOfTheta(self, par):
        theta = deepcopy(par['m'])

        for d in range(par['D']):
            if sum(theta[d]) == 0:
                theta[d] = np.asarray([1.0 / len(theta[d]) for _ in range(len(theta[d]))])
            else:
                theta[d] = np.asarray(theta[d])
                theta[d] = 1.0 * theta[d] / sum(theta[d])

        return np.matrix(theta)

    def ComputePosteriorEstimateOfPhi(self, par):
        phi = deepcopy(par['n'])

        for t in range(par['T']):
            if sum(phi[t]) == 0:
                phi[t] = np.asarray([1.0 / len(phi[t]) for _ in range(len(phi[t]))])
            else:
                phi[t] = np.asarray(phi[t])
                phi[t] = 1.0 * phi[t] / sum(phi[t])

        return np.matrix(phi)

    def TopicsOverTimeGibbsSampling(self, par):
        for iteration in range(par['max_iterations']):
            for d in range(par['D']):
                for i in range(par['N'][d]):
                    word_di = par['w'][d][i]
                    t_di = par['t'][d][i]
                    # if t_di == 0:
                    #     t_di == 1e-6
                    # if t_di == 1:
                    #     t_di == 1 - 1e-6
                    old_topic = par['z'][d][i]
                    par['m'][d][old_topic] -= 1
                    par['n'][old_topic][word_di] -= 1
                    par['n_sum'][old_topic] -= 1

                    topic_probabilities = []
                    for topic_di in range(par['T']):
                        psi_di = par['psi'][topic_di]
                        topic_probability = 1.0 * (par['m'][d][topic_di] + par['alpha'][topic_di])
                        try:
                            xx = math.pow(t_di, psi_di[0] - 1) * math.pow(1 - t_di, psi_di[1] - 1)
                        except Exception as e:
                            print(e, psi_di, t_di)
                        topic_probability *= xx*0.45
                        yy = float(par['betafunc_psi'][topic_di])
                        if yy == 0:
                            yy = 1e-77
                        topic_probability /= yy
                        topic_probability *= par['n'][topic_di][word_di] + par['beta'][word_di]
                        zz = par['n_sum'][topic_di] + par['beta_sum']
                        if zz == 0:
                            zz == 1e-77
                        topic_probability /= float(par['n_sum'][topic_di] + par['beta_sum'])
                        if np.isnan(topic_probability):
                            pass
                            # print((par['n'][topic_di][word_di] + par['beta'][word_di]), zz, xx, yy,par['m'][d][topic_di] + par['alpha'][topic_di])
                        topic_probabilities.append(topic_probability)
                    sum_topic_probabilities = sum(topic_probabilities)
                    if sum_topic_probabilities == 0:
                        topic_probabilities = [1.0 / par['T'] for _ in range(par['T'])]
                        sum_topic_probabilities = 1
                        # print(sum_topic_probabilities)
                    else:
                        topic_probabilities = [p / sum_topic_probabilities for p in topic_probabilities]

                    try:
                        new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
                    except Exception as e:
                        # print(sum_topic_probabilities)
                        # print(np.random.multinomial(1, topic_probabilities, size=1))
                        # print(topic_probabilities)
                        # print(type(topic_probabilities))
                        pass
                    par['z'][d][i] = new_topic
                    par['m'][d][new_topic] += 1
                    par['n'][new_topic][word_di] += 1
                    par['n_sum'][new_topic] += 1

                if d % 500 == 0:
                    # print('Done with iteration {iteration} and document {document}'.format(iteration=iteration,document=d))
                    pass
            par['psi'] = self.GetMethodOfMomentsEstimatesForPsi(par)
            par['betafunc_psi'] = [scipy.special.beta(par['psi'][t][0], par['psi'][t][1]) for t in range(par['T'])]
        par['theta'], par['phi'] = self.ComputePosteriorEstimatesOfThetaAndPhi(par)
        # print(par['V'])
        return par['theta'], par['phi'], par['psi']

    def ContinueGibbsSampling(self, par):
        for iteration in range(par['max_iterations']):
            for d in range(par['D']):
                for i in range(par['N'][d]):
                    word_di = par['w'][d][i]
                    t_di = par['t'][d][i]
                    # if t_di == 0:
                    #     t_di == 1e-6
                    # if t_di == 1:
                    #     t_di == 1 - 1e-6
                    old_topic = par['z'][d][i]
                    par['m'][d][old_topic] -= 1
                    par['n'][old_topic][word_di] -= 1
                    par['n_sum'][old_topic] -= 1

                    topic_probabilities = []
                    for topic_di in range(par['T']):
                        psi_di = par['psi'][topic_di]
                        topic_probability = 1.0 * (par['m'][d][topic_di] + par['alpha'][topic_di])
                        try:
                            xx = math.pow(t_di, psi_di[0] - 1) * math.pow(1 - t_di, psi_di[1] - 1)
                        except Exception as e:
                            # print(e, psi_di, t_di)
                            pass
                        topic_probability *= xx*0.25
                        topic_probability *= par['rank'][topic_di][word_di]
                        yy = float(par['betafunc_psi'][topic_di])
                        if yy == 0:
                            yy = 1e-77
                        topic_probability /= yy
                        topic_probability *= par['n'][topic_di][word_di] + par['beta'][word_di]
                        topic_probability /= float(par['n_sum'][topic_di] + par['beta_sum'])
                        topic_probabilities.append(topic_probability)
                    sum_topic_probabilities = sum(topic_probabilities)
                    if sum_topic_probabilities == 0:
                        topic_probabilities = [1.0 / par['T'] for _ in range(par['T'])]
                        sum_topic_probabilities = 1
                    else:
                        topic_probabilities = [p / sum_topic_probabilities for p in topic_probabilities]

                    try:
                        new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
                    except Exception as e:
                        # print(sum_topic_probabilities)
                        # print(np.random.multinomial(1, topic_probabilities, size=1))
                        # print(topic_probabilities)
                        # print(type(topic_probabilities))
                        pass
                    par['z'][d][i] = new_topic
                    par['m'][d][new_topic] += 1
                    par['n'][new_topic][word_di] += 1
                    par['n_sum'][new_topic] += 1

                if d % 500 == 0:
                    # print('Done with iteration {iteration} and document {document}'.format(iteration=iteration,document=d))
                    pass
            par['psi'] = self.GetMethodOfMomentsEstimatesForPsi(par)
            par['betafunc_psi'] = [scipy.special.beta(par['psi'][t][0], par['psi'][t][1]) for t in range(par['T'])]
        self.CalculateCounts(par)
        par['theta'], par['phi'] = self.ComputePosteriorEstimatesOfThetaAndPhi(par)
        return par['theta'], par['phi'], par['psi']

    def save(self, file_name, par):
        """
        :param file_name:文件保存路径 + 名称
        :param par:
        :return:
        """
        # 保存文件
        tot_pickle = open(file_name, 'wb')
        pickle.dump(par, tot_pickle)
        tot_pickle.close()
        pass

    def load(self, file_name):
        # 加载文件
        f = open(file_name, 'rb')
        par = pickle.loads(f.read())
        f.close()
        return par

    def SemiSupervisedInitialize(self, par, num, bad_list=[], topic_mark=[],
                   good_words_rank = 300, bad_words_rank = 0.001, goodwords=[], badwords=[], TopicKey = True):
        """
        :bad_list ： 删除的主题
        :topic_mark:
        :TopicKey:  True 彻底删除bad_list内主题 false 重新训练bad_list内主题
        :return:
        """
        goodwords = goodwords if goodwords else [[] for _ in range(par['T'])]
        badwords = badwords if badwords else [[] for _ in range(par['T'])]
        if goodwords or badwords:
            for topic in range(par['T']):
                for word in goodwords[topic]:
                    par['rank'][topic][par['word_id'][word]] = good_words_rank
                for word in badwords[topic]:
                    par['rank'][topic][par['word_id'][word]] = bad_words_rank
        par['max_iterations'] = num
        if TopicKey:
            remain_topic = [topic for topic in list(set(topic_mark)) if topic not in bad_list]
        else:
            remain_topic = list(set(topic_mark))
        # print(remain_topic)
        if bad_list and topic_mark:
            for d in range(par['D']):
                for w in range(len(par['z'][d])):
                    if par['z'][d][w] not in remain_topic:
                        par['z'][d][w] = random.sample(remain_topic,1)[0]
            par['m'] = [[0 for t in range(par['T'])] for d in range(par['D'])]
            par['n'] = [[0 for v in range(par['V'])] for t in range(par['T'])]
            self.CalculateCounts(par)
            par['psi'] = self.GetMethodOfMomentsEstimatesForPsi(par)
            par['betafunc_psi'] = [scipy.special.beta(par['psi'][t][0], par['psi'][t][1]) for t in range(par['T'])]
            par['theta'], par['phi'] = self.ComputePosteriorEstimatesOfThetaAndPhi(par)
        return par

    def SetEntityRank(self, par, entity_words_id=[], entity_rank = 1.5):
        if entity_words_id:
            for id in range(par['V']):
                if entity_words_id[id] == 1:
                    for i in range(len(par['T'])):
                        par['rank'][i][id] = entity_rank
        return par



