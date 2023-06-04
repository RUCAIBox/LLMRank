import os
import random
import argparse
import numpy as np
import torch
from logging import getLogger
from recbole.config import Config
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger
from recbole.data import data_preparation

from utils import get_model


def sample_candidates(dataset_name, strategy, n_users, n_cands, **kwargs):
    if strategy == 'random':
        model_name = 'SASRec'
    elif strategy == 'bm25':
        model_name = 'BM25'
    else:
        raise NotImplementedError()
    model_class = get_model(model_name)

    # configurations initialization
    props = ['props/overall.yaml', f'props/{model_name}.yaml', f'props/{dataset_name}.yaml']
    print(props)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = SequentialDataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data._dataset).to(config['device'])

    # sample selected users
    selected_users = random.sample(dataset.field2id_token['user_id'][1:].tolist(), n_users)
    selected_uids = np.array([dataset.field2token_id['user_id'][_] for _ in selected_users])
    rand_user_file = os.path.join(f'{config["data_path"]}/{config["dataset"]}.{strategy}')
    with open(rand_user_file, 'w', encoding='utf-8') as file:
        if strategy == 'random':
            for user in selected_users:
                selected_items = random.sample(dataset.field2id_token['item_id'][1:].tolist(), n_cands)
                file.write(f'{user}\t{" ".join(selected_items)}\n')
        elif strategy == 'bm25':
            def get_user_text_list():
                batch_user_ids = test_data.dataset.inter_feat['user_id'].numpy().tolist()
                user_text_list = []
                for i in range(selected_uids.shape[0]):
                    pr = batch_user_ids.index(selected_uids[i])
                    his_iids = [_ for _ in test_data.dataset.inter_feat['item_id_list'][pr].numpy().tolist() if _ != 0]
                    user_text = []
                    for _ in his_iids:
                        user_text = user_text + model.encoded_item_text[_]
                    user_text_list.append(user_text)
                return user_text_list

            user_text_list = get_user_text_list()
            average_idf = sum(map(lambda k: float(model.bm25_model.idf[k]), model.bm25_model.idf.keys())) / len(model.bm25_model.idf.keys())

            all_item_score = []
            for u_text in user_text_list:
                scores = model.bm25_model.get_scores(u_text, average_idf)
                all_item_score.append(scores)

            all_item_score = torch.from_numpy(np.array(all_item_score))
            recall_items = torch.topk(all_item_score, n_cands)[1].numpy()

            assert len(selected_users) == recall_items.shape[0]
            for i, user in enumerate(selected_users):
                selected_items = [dataset.field2id_token['item_id'][_] for _ in recall_items[i].tolist()]
                file.write(f'{user}\t{" ".join(selected_items)}\n')
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-s', type=str, default='random', help='strategy to sample negative items')
    parser.add_argument('-u', type=int, default=200, help='number of selected users')
    parser.add_argument('-k', type=int, default=100, help='number of recalled candidates for each user')
    args, unparsed = parser.parse_known_args()
    print(args)

    sample_candidates(args.d, args.s, args.u, args.k)
