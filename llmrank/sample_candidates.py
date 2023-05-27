import argparse
from logging import getLogger
from recbole.config import Config
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger
from recbole.data import data_preparation
import random
import os

from utils import get_model


def sample_candidates(dataset_name, strategy, **kwargs):
    if strategy == 'random':
        model_name = 'SASRec'
    else:
        raise NotImplementedError()
    model_class = get_model(model_name)

    # configurations initialization
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'props/overall.yaml']
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

    # sample selected users
    selected_users = random.sample(dataset.field2id_token['user_id'][1:].tolist(), 200)
    rand_user_file = os.path.join(f'{config["data_path"]}/{config["dataset"]}.{strategy}')
    with open(rand_user_file, 'w', encoding='utf-8') as file:
        if strategy == 'random':
            for user in selected_users:
                selected_items = random.sample(dataset.field2id_token['item_id'][1:].tolist(), 100)
                file.write(f'{user}\t{" ".join(selected_items)}\n')
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('-s', type=str, default='random', help='strategy to sample negative items')
    args, unparsed = parser.parse_known_args()
    print(args)

    sample_candidates(args.d, args.s)
