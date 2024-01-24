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
from recbole.utils.case_study import full_sort_topk

from utils import get_model


def write_sampled_candidates_to_file(selected_users, recall_items, dataset, file):
    assert len(selected_users) == recall_items.shape[0]
    for i, user in enumerate(selected_users):
        selected_items = [dataset.field2id_token['item_id'][_] for _ in recall_items[i].tolist()]
        file.write(f'{user}\t{" ".join(selected_items)}\n')


def sample_candidates(dataset_name, strategy, n_users, n_cands, **kwargs):
    strategy2model_name = {
        'random': 'SASRec',
        'bm25': 'BM25',
        'bert': 'Rank',
        'pop': 'Pop',
        'bpr': 'BPR',
        'gru4rec': 'GRU4Rec',
        'sasrec': 'SASRec'
    }
    assert strategy in strategy2model_name, f'strategy [{strategy}] does not exist.'
    model_name = strategy2model_name[strategy]
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
    if model_name in ['Pop', 'BPR', 'GRU4Rec', 'SASRec']:
        chpt_path = f'pretrained_models/{model_name}-{dataset_name}.pth'
        checkpoint = torch.load(chpt_path, map_location=config['device'])
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        logger.info("Loading model parameters from {}".format(chpt_path))

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

            write_sampled_candidates_to_file(selected_users, recall_items, dataset, file)
        elif strategy == 'bert':
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            bert = AutoModel.from_pretrained('bert-base-uncased').to(config['device'])

            def get_user_text_list():
                batch_user_ids = test_data.dataset.inter_feat['user_id'].numpy().tolist()
                user_text_list = []
                for i in range(selected_uids.shape[0]):
                    pr = batch_user_ids.index(selected_uids[i])
                    his_iids = [_ for _ in test_data.dataset.inter_feat['item_id_list'][pr].numpy().tolist() if _ != 0]
                    user_text = [model.item_text[_] for _ in his_iids]
                    user_text_list.append(' '.join(user_text))
                return user_text_list

            def get_bert_results(encode_text):
                text_emb = []
                batch_size = 128
                attn_mask = torch.split(encode_text['attention_mask'], batch_size, dim=0)
                encode_ids = torch.split(encode_text['input_ids'], batch_size, dim=0)
                for index, ids in enumerate(encode_ids):
                    input_id = ids.to(config['device'])
                    mask = attn_mask[index].to(config['device'])
                    with torch.no_grad():
                        output_tuple = bert(
                            input_id,
                            attention_mask=mask
                        )
                    output = output_tuple[0][:,0,:].detach().cpu()
                    text_emb.append(output)
                return torch.cat(text_emb, dim=0)

            def bert_encode_text(text_input):
                token_text = tokenizer.batch_encode_plus(
                    text_input,
                    max_length=512,
                    truncation=True,
                    padding='longest',
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                text_emb = get_bert_results(token_text)
                return text_emb

            user_text_list = get_user_text_list()

            user_text_emb = bert_encode_text(user_text_list).to(config['device'])
            item_text_emb = bert_encode_text(model.item_text).to(config['device'])

            user_item_sim = torch.matmul(user_text_emb.to(config['device']), item_text_emb.transpose(0, 1).to(config['device']))
            recall_items = torch.topk(user_item_sim, n_cands)[1].cpu().numpy()

            write_sampled_candidates_to_file(selected_users, recall_items, dataset, file)
        elif strategy in ['pop', 'bpr', 'gru4rec', 'sasrec']:
            score, recall_items = \
                full_sort_topk(selected_uids, model=model, test_data=test_data, k=n_cands, device=config['device'])
            recall_items = recall_items.cpu().numpy()
            write_sampled_candidates_to_file(selected_users, recall_items, dataset, file)
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
