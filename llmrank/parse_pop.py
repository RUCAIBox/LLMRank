import argparse
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
import os.path as osp
from utils import get_model
import numpy as np

def load_text(config, dataset):
    token_text = {}
    item_text = ['[PAD]']
    feat_path = osp.join(config['data_path'], f"{dataset.dataset_name}.item")
    id_token = dataset.field2id_token['item_id']
    if dataset.dataset_name in ['ml-1m', 'ml-1m-full']:
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                item_id, movie_title, release_year, genre = line.strip().split('\t')
                token_text[item_id] = movie_title
        for i, token in enumerate(id_token):
            if token == '[PAD]': continue
            raw_text = token_text[token]
            if raw_text.endswith(', The'):
                raw_text = 'The ' + raw_text[:-5]
            elif raw_text.endswith(', A'):
                raw_text = 'A ' + raw_text[:-3]
            item_text.append(raw_text)
        return item_text
    elif dataset.dataset_name in ['Games', 'Games-6k']:
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                item_id, title = line.strip().split('\t')
                token_text[item_id] = title
        for i, token in enumerate(id_token):
            if token == '[PAD]': continue
            raw_text = token_text[token]
            item_text.append(raw_text)
        return item_text
    else:
        raise NotImplementedError()

def load_pop(dataset, item_text):
    item_counter = dataset.item_counter
    item_title_pop = {}
    for item_id, item_pop in item_counter.items():
        item_title = item_text[item_id]
        if item_title.endswith(', The'):
            item_title = 'The ' + item_title[:-5]
        elif item_title.endswith(', A'):
            item_title = 'A ' + item_title[:-3]
        item_title_pop[item_title] = item_pop
    return item_title_pop


def parse(model_name, dataset_name,log_path, **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'openai_api.yaml', 'props/overall.yaml']
    print(props)
    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)

    # dataset filtering
    if model_name == 'UniSRec':
        from dataset import UniSRecDataset
        dataset = UniSRecDataset(config)
    elif model_name == 'VQRec':
        from dataset import VQRecDataset
        dataset = VQRecDataset(config)
    else:
        dataset = SequentialDataset(config)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    item_text = load_text(config,train_data.dataset)
    item_title_pop = load_pop(dataset,item_text)
    uid = 0
    user_reco_pop = {}
    user_reco_pop[0] = [0] * config['recall_budget']
    with open(log_path,'rt') as f:
        candidate_text = None
        for line in f:
            line = line.strip()
            if dataset_name in ['ml-1m', 'ml-1m-full']:
                if "Here are answer: " in line:
                    try:
                        ans_movies = eval(line[line.index("Here are answer: ") + len("Here are answer: "):])
                    except:
                        print("answer of LLM is unmatched")
                        continue
                    uid += 1
                    user_reco_pop[uid] = [0] * config['recall_budget']
                    for i, order_movie in enumerate(ans_movies):
                        pr = order_movie.find('. ')
                        if order_movie[:pr].isdigit():
                            movie_name = order_movie[pr + 2:]
                        else:
                            movie_name = order_movie
                        if i >= config['recall_budget'] or movie_name not in item_title_pop: continue
                        user_reco_pop[uid][i] = item_title_pop[movie_name]
            elif dataset_name in ['Games', 'Games-6k']:
                if "Here are candidates: " in line:
                    candidate_text = eval(line[line.index("Here are candidates: ") + len("Here are candidates: "):])
                if "Here are answer: " in line:
                    try:
                        ans_order = [int(order) for order in eval(line[line.index("Here are answer: ") + len("Here are answer: "):])]
                    except:
                        print("answer of LLM is unmatched")
                        continue
                    uid += 1
                    user_reco_pop[uid] = [0] * config['recall_budget']
                    for i, order in enumerate(ans_order):
                        if i >= config['recall_budget']: continue
                        user_reco_pop[uid][i] = item_title_pop[candidate_text[order]]

    pos_pop = [[] for _ in range(config['recall_budget'])]
    for reco_pop_list in user_reco_pop.values():
        for i in range(len(reco_pop_list)):
            if reco_pop_list[i] > 0:  # 未匹配上的item不参与平均值计算
                pos_pop[i].append(reco_pop_list[i])
    y = []
    for pop in pos_pop:
        y.append(np.mean(pop))
    print(y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="Rank", help="model name")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-lp', type=str, default='log/Rank/Rank-ml-1m-Jun-07-2023_13-09-03-3c1e76.log', help='log path')
    args, unparsed = parser.parse_known_args()
    print(args)

    parse(args.m, args.d,args.lp)
