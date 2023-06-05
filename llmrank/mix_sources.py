import os
import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger
from recbole.data import data_preparation

from utils import get_model


def load_single_source(config, source_name):
    cur_data = []
    filepath = os.path.join(config['data_path'], f'{config["dataset"]}.{source_name}')
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            user_token, candidate_item_token = line.strip().split('\t')
            candidate_item_token_list = candidate_item_token.split(' ')
            cur_data.append([user_token, candidate_item_token_list])
    return cur_data


def mix_source(model_name, dataset_name, sources, k):
    props = ['props/overall.yaml', f'props/{model_name}.yaml', f'props/{dataset_name}.yaml']
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = SequentialDataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    sources = sources.split(',')

    all_data = []
    for source_name in sources:
        all_data.append(load_single_source(config, source_name))

    all_selected_items = []
    batch_user_ids = test_data.dataset.inter_feat['user_id'].numpy().tolist()
    ground_truth_items = test_data.dataset.inter_feat['item_id']
    for i in range(len(all_data[0])):
        for j in range(len(sources)):
            assert all_data[j][i][0] == all_data[0][i][0]

        user_token = all_data[0][i][0]
        user_id = dataset.field2token_id['user_id'][user_token]
        pr = batch_user_ids.index(user_id)
        ground_truth_item_id = ground_truth_items[pr].item()
        ground_truth_item_token = dataset.field2id_token['item_id'][ground_truth_item_id]

        cur_selected_items = []
        for j in range(len(sources)):
            lr = cnt = 0
            while cnt < k:
                cur_item = all_data[j][i][1][lr]
                lr += 1
                cur_selected_items.append(cur_item)
                cnt += 1
        assert len(cur_selected_items) == k * len(sources)
        all_selected_items.append([user_token, ' '.join(cur_selected_items)])
    
    short_name = ''.join([_[:2] for _ in sources])
    print(f'output {short_name}')
    output_user_file = os.path.join(config['data_path'], f'{config["dataset"]}.{short_name}_{k}')
    with open(output_user_file, 'w', encoding='utf-8') as file:
        for user, item_list in all_selected_items:
            file.write(f'{user}\t{item_list}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="SASRec", help="name of models")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-s', type=str, default='random,bm25,bert,pop,bpr,gru4rec,sasrec', help='strategy to sample negative items')
    parser.add_argument('-k', type=int, default=3, help='#items from each source')
    args, unparsed = parser.parse_known_args()
    print(args)

    mix_source(args.m, args.d, args.s, args.k)
