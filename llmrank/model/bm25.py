import os.path as osp
import torch
from recbole.model.abstract_recommender import SequentialRecommender
import numpy as np
import math

from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from six import iteritems
from six.moves import xrange


# BM25 parameters.
PARAM_K1 = 1.5
PARAM_B = 0.75
EPSILON = 0.25


class BM25_Model(object):
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), corpus)) / self.corpus_size
        self.corpus = corpus
        self.f = []
        self.df = {}
        self.idf = {}
        self.initialize()

    def initialize(self):
        for document in self.corpus:
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.f.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)

    def get_score(self, document, index, average_idf):
        score = 0
        for word in document:
            if word not in self.f[index]:
                continue
            idf = self.idf[word] if self.idf[word] >= 0 else EPSILON * average_idf
            score += (idf * self.f[index][word] * (PARAM_K1 + 1)
                      / (self.f[index][word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.corpus_size / self.avgdl)))
        return score

    def get_scores(self, document, average_idf):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(document, index, average_idf)
            scores.append(score)
        return scores


class BM25(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.max_his_len = config['max_his_len']
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id']
        self.item_text = self.load_text()

        self.fake_fn = torch.nn.Linear(1, 1)
        self.encoded_item_text = self.load_segment_text(self.item_text)
        self.bm25_model = BM25_Model(self.encoded_item_text)

    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        if self.dataset_name in ['ml-1m', 'ml-1m-full']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name in ['Games', 'Games-6k']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    try:
                        item_id, title = line.strip().split('\t')
                    except:
                        print(line)
                        exit(1)
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        else:
            raise NotImplementedError()

    @staticmethod
    def load_segment_text(input_text):
        all_text = []
        stopWords = set(stopwords.words('english'))

        for row in input_text:
            sentence = word_tokenize(row)
            sentence = [w for w in sentence if w not in stopWords]
            all_text.append(sentence)

        return all_text

    def predict_on_subsets(self, interaction, idxs):
        """
        :param interaction:
        :param idxs: item id retrieved by candidate generation models [batch_size, candidate_size]
        :return:
        """

        user_id = interaction[self.USER_ID]
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]

        user_text_list = []
        for i in tqdm(range(user_id.shape[0])):
            real_his_len = min(self.max_his_len, user_his_len[i].item())
            user_his_text = []
            for j in range(real_his_len):
                user_his_text += self.item_text[user_his[i, user_his_len[i].item() - real_his_len + j].item()]
            user_text_list.append(user_his_text)

        average_idf = sum(map(lambda k: float(self.bm25_model.idf[k]), self.bm25_model.idf.keys())) / len(self.bm25_model.idf.keys())

        all_item_score = []
        for u_text in user_text_list:
            cur_scores = self.bm25_model.get_scores(u_text, average_idf)
            all_item_score.append(cur_scores)

        all_item_score = torch.from_numpy(np.array(all_item_score))

        scores = torch.full((user_id.shape[0], self.n_items), -10000.)
        for i in range(idxs.shape[0]):
            for j in range(idxs.shape[1]):
                scores[i, idxs[i, j]] = all_item_score[i, idxs[i, j]]

        return scores
