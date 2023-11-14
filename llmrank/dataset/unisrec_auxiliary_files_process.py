import os
from transformers import AutoTokenizer, AutoModel
import argparse
import torch
import numpy as np

def generate_item_embedding(args, item_text_list, tokenizer, model,):
    print(f'Generate Text Embedding by {args.emb_type}: ')
    print(' Dataset: ', args.dataset)

    items, texts = zip(*item_text_list)
    item2row = {}
    for row, item in enumerate(items):
        item2row[int(item)] = row
    
    dict_file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '_item_dataset2row.npy')
    np.save(dict_file, item2row)   
    embeddings = []
    start, batch_size = 0, args.batch_size
    while start < len(texts):
        sentences = texts[start: start + batch_size]
        encoded_sentences = tokenizer(sentences, padding=True, max_length=512,
                                      truncation=True, return_tensors='pt').to(args.device)
        outputs = model(**encoded_sentences)
        if args.emb_type == 'CLS':
            cls_output = outputs.last_hidden_state[:, 0, ].detach().cpu()
            embeddings.append(cls_output)
        elif args.emb_type == 'Mean':
            masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
            mean_output = masked_output[:,1:,:].sum(dim=1) / \
                encoded_sentences['attention_mask'][:,1:].sum(dim=-1, keepdim=True)
            mean_output = mean_output.detach().cpu()
            embeddings.append(mean_output)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.output_path, args.dataset,
                        args.dataset + '.feat1' + args.emb_type)
    embeddings.tofile(file)


def load_item_text_list(args,):
    # dataset_full_name = amazon_dataset2fullname[args.dataset] if args.dataset != 'ml-1m' else args.dataset
    item_text_list = []
    with open(os.path.join(args.input_path, args.dataset, args.dataset + '.item')) as f:
        f.readline()
        for line in f:
            line = line.strip().split('\t')
            item_text_list.append((int(line[0]), line[1]))
    return item_text_list

def load_plm(model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    else:
        return torch.device(
            'cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Games', help='Pantry / Scientific / Instruments / Arts / Office')
    parser.add_argument('--input_path', type=str, default='dataset/')
    parser.add_argument('--output_path', type=str, default='dataset/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--max_length', type=int, default=512, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='bert-base-uncased')
    parser.add_argument('--emb_type', type=str, default='CLS', help='item text emb type, can be CLS or Mean')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    item_text_list = load_item_text_list(args)
    device = set_device(args.gpu_id)
    args.device = device
    plm_tokenizer, plm_model = load_plm(args.plm_name)
    plm_model = plm_model.to(device)
    generate_item_embedding(args,item_text_list,plm_tokenizer,plm_model)
