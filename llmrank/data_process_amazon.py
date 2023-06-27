import argparse
import collections
import gzip
import json
import os
from tqdm import tqdm
import numpy as np
from utils import check_path, amazon_dataset2fullname


def load_ratings(file):
    users, items, inters = set(), set(), set()
    with open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                users.add(user)
                items.add(item)
                inters.add((user, item, float(rating), int(time)))
            except ValueError:
                print(line)
    return users, items, inters
    

def load_meta_items(file):
    items = {}
    with gzip.open(file, 'r') as fp:
        for line in tqdm(fp, desc='Load metas'):
            data = json.loads(line)
            item = data['asin']
            title = data['title'].replace('\t',' ')
            if title == '':
                continue
            brand = data['brand']
            category = data['category']
            items[item] = {'title':title,'brand': brand, 'category': category}
    return items


def get_user2count(inters):
    user2count = collections.defaultdict(int)
    for unit in inters:
        user2count[unit[0]] += 1
    return user2count


def get_item2count(inters):
    item2count = collections.defaultdict(int)
    for unit in inters:
        item2count[unit[1]] += 1
    return item2count


def generate_candidates(unit2count, threshold):
    cans = set()
    for unit, count in unit2count.items():
        if count >= threshold:
            cans.add(unit)
    return cans, len(unit2count) - len(cans)


def filter_inters(inters, can_items=None,
                  user_k_core_threshold=0, item_k_core_threshold=0):
    new_inters = []

    # filter by meta items
    if can_items:
        print('\nFiltering by meta items: ')
        for unit in inters:
            if unit[1] in can_items.keys():
                new_inters.append(unit)
        inters, new_inters = new_inters, []
        print('    The number of inters: ', len(inters))

    # filter by k-core
    if user_k_core_threshold or item_k_core_threshold:
        print('\nFiltering by k-core:')
        idx = 0
        user2count = get_user2count(inters)
        item2count = get_item2count(inters)

        while True:
            new_user2count = collections.defaultdict(int)
            new_item2count = collections.defaultdict(int)
            users, n_filtered_users = generate_candidates( # users is set
                user2count, user_k_core_threshold)
            items, n_filtered_items = generate_candidates(
                item2count, item_k_core_threshold)
            if n_filtered_users == 0 and n_filtered_items == 0:
                break
            for unit in inters:
                if unit[0] in users and unit[1] in items:
                    new_inters.append(unit)
                    new_user2count[unit[0]] += 1
                    new_item2count[unit[1]] += 1
            idx += 1
            inters, new_inters = new_inters, []
            user2count, item2count = new_user2count, new_item2count
            print('    Epoch %d The number of inters: %d, users: %d, items: %d'
                    % (idx, len(inters), len(user2count), len(item2count)))
    return inters


def make_inters_in_order(inters):
    user2inters, new_inters = collections.defaultdict(list), list()
    for inter in inters:
        user, item, rating, timestamp = inter
        user2inters[user].append((user, item, rating, timestamp))
    for user in user2inters:
        user_inters = user2inters[user]
        user_inters.sort(key=lambda d: d[3])
        interacted_item = set()
        for inter in user_inters: 
            if inter[1] in interacted_item:  # deduplicate
                continue
            interacted_item.add(inter[1])
            new_inters.append(inter)
    return new_inters


def preprocess_rating(args):
    dataset_full_name = amazon_dataset2fullname[args.dataset]

    print('Process rating data: ')
    print(' Dataset: ', args.dataset)

    # load ratings
    rating_file_path = os.path.join(args.input_path, 'Ratings', dataset_full_name + '.csv')
    rating_users, rating_items, rating_inters,  = load_ratings(rating_file_path)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.input_path, 'Metadata', f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)

    # 1. Filter items w/o meta data;
    # 2. K-core filtering;
    print('The number of raw inters: ', len(rating_inters))

    rating_inters = make_inters_in_order(rating_inters)

    rating_inters = filter_inters(rating_inters, can_items=meta_items,
                                  user_k_core_threshold=args.user_k,
                                  item_k_core_threshold=args.item_k)

    # sort interactions chronologically for each user
    rating_inters = make_inters_in_order(rating_inters)
    print('\n')

    # return: list of (user_ID, item_ID, rating, timestamp)
    return rating_inters, meta_items, 



def write_text_file(item_texts, file):
    print('Writing text file: ')
    with open(file, 'w') as f:
        f.write('item_id:token\ttitle:token_seq\n')
        for item, texts in item_texts.items():
            title = texts['title']
            f.write(str(item)+'\t'+title+'\n')

def convert_inters2dict(inters):
    user2items = collections.defaultdict(list)
    user2index, item2index = dict(), dict()
    for inter in inters:
        user, item, rating, timestamp = inter
        if user not in user2index:
            user2index[user] = len(user2index)
        if item not in item2index:
            item2index[item] = len(item2index)
        user2items[user2index[user]].append(item2index[item])
    return user2items, user2index, item2index


def generate_indexed_data(args, rating_inters):
    print('Split dataset: ')
    print(' Dataset: ', args.dataset)
    user2items, user2index, item2index = convert_inters2dict(rating_inters)
    all_inters = dict()
    for u_index in range(len(user2index)):
        all_inters[u_index]= [str(item) for item in user2items[u_index]]
    return all_inters, user2items, user2index, item2index, 


def load_unit2index(file):
    unit2index = dict()
    with open(file, 'r') as fp:
        for line in fp:
            unit, index = line.strip().split('\t')
            unit2index[unit] = int(index)
    return unit2index


def write_remap_index(unit2index, file):
    with open(file, 'w') as fp:
        for unit in unit2index:
            fp.write(unit + '\t' + str(unit2index[unit]) + '\n')


def convert_to_atomic_files(args, all_data):
    print('Convert dataset: ')
    print(' Dataset: ', args.dataset)
    uid_list = list(all_data.keys())
    uid_list.sort(key=lambda t: int(t))

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}_id.inter'), 'w') as file:
        file.write('user_id:token\titem_id_list:token_seq\n')
        for uid in uid_list:
            item_seq = all_data[uid]
            file.write(f'{uid}\t{" ".join(item_seq)}\n')

    with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.train.inter'), 'w') as f1:
        with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.valid.inter'), 'w') as f2:
            with open(os.path.join(args.output_path, args.dataset, f'{args.dataset}.test.inter'), 'w') as f3:
                f1.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
                f2.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
                f3.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
                for user, item_list in all_data.items():
                    test_target_item = item_list[-1]
                    test_list = item_list[:-1][-args.max_length:]
                    f3.write(str(user)+'\t'+' '.join([str(item) for item in test_list])+'\t'+str(test_target_item)+'\n')
                    valid_target_item = item_list[-2]
                    valid_list = item_list[:-2][-args.max_length:]
                    f2.write(str(user)+'\t'+' '.join([str(item) for item in valid_list])+'\t'+str(valid_target_item)+'\n')
                    train_list = item_list[:-2]
                    for i in range(1,len(train_list)):
                        sub_item_list = train_list[:-i][-args.max_length:]
                        sub_target_item = train_list[-i]
                        f1.write(str(user)+'\t'+' '.join([str(item) for item in sub_item_list])+'\t'+str(sub_target_item)+'\n')
                    f1.write(str(user)+'\t'+' '.join([str(item) for item in train_list])+'\t'+str(train_target_item)+'\n')





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CDs', help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--user_k', type=int, default=5, help='user k-core filtering')
    parser.add_argument('--item_k', type=int, default=5, help='item k-core filtering')
    parser.add_argument('--input_path', type=str, default='dataset/raw/')
    parser.add_argument('--output_path', type=str, default='dataset/')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--max_length', type=int, default=50, help='ID of running GPU')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # load interactions from raw rating file
    rating_inters, meta_items, = preprocess_rating(args)

    all_inters, user2items, user2index, item2index = \
        generate_indexed_data(args, rating_inters)

    check_path(os.path.join(args.output_path, args.dataset))

    # save interaction sequences into atomic files
    convert_to_atomic_files(args,all_inters)
    item2text = collections.defaultdict(dict)
    for item, item_id in item2index.items():
        item2text[item_id] = meta_items[item]
    
    write_text_file(item2text, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item'))
    write_remap_index(user2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.user2index'))
    write_remap_index(item2index, os.path.join(args.output_path, args.dataset, f'{args.dataset}.item2index'))
