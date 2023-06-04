import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset


class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.plm_size = config['plm_size']
        self.plm_suffix = config['plm_suffix']
        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight)

    def load_plm_embedding(self):
        feat_path = osp.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)
        print(loaded_feat.shape)

        mapped_feat = np.zeros((self.item_num, self.plm_size))
        item2row_path = osp.join(self.config['data_path'], f'{self.dataset_name}_item_dataset2row.npy')
        item2row = np.load(item2row_path,allow_pickle=True).item()
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat[i] = loaded_feat[item2row[int(token)]]
        return mapped_feat

    def weight2emb(self, weight):
        plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding


class VQRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.index_suffix = config['index_suffix']
        self.pq_codes = self.load_index()

    def load_index(self):
        import faiss
        if self.config['index_pretrain_dataset'] is not None:
            index_dataset = self.config['index_pretrain_dataset']
        else:
            index_dataset = self.dataset_name
        index_path = os.path.join(
            self.config['index_path'],
            index_dataset,
            f'{index_dataset}.{self.index_suffix}'
        )
        self.logger.info(f'Index path: {index_path}')
        uni_index = faiss.read_index(index_path)
        old_pq_codes, _, _, _ = self.parse_faiss_index(uni_index)
        old_code_num = old_pq_codes.shape[0]

        self.plm_suffix = self.config['plm_suffix']
        self.plm_size = self.config['plm_size']
        feat_path = os.path.join(self.config['data_path'], f'{self.dataset_name}.{self.plm_suffix}')
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(-1, self.plm_size)

        uni_index.add(loaded_feat)
        all_pq_codes, centroid_embeds, coarse_embeds, opq_transform = self.parse_faiss_index(uni_index)
        pq_codes = all_pq_codes[old_code_num:]
        assert self.code_dim == pq_codes.shape[1], pq_codes.shape
        # assert self.item_num == 1 + pq_codes.shape[0], pq_codes.shape

        # uint8 -> int32 to reserve 0 padding
        pq_codes = pq_codes.astype(np.int32)
        # 0 for padding
        pq_codes = pq_codes + 1
        # flatten pq codes
        base_id = 0
        for i in range(self.code_dim):
            pq_codes[:, i] += base_id
            base_id += self.code_cap + 1

        mapped_codes = np.zeros((self.item_num, self.code_dim), dtype=np.int32)
        item2row_path = osp.join(self.config['data_path'], f'{self.dataset_name}_item_dataset2row.npy')
        item2row = np.load(item2row_path, allow_pickle=True).item()
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_codes[i] = pq_codes[item2row[int(token)]]
            
        self.plm_embedding = torch.FloatTensor(loaded_feat)
        return torch.LongTensor(mapped_codes)

    @staticmethod
    def parse_faiss_index(pq_index):
        import faiss
        vt = faiss.downcast_VectorTransform(pq_index.chain.at(0))
        assert isinstance(vt, faiss.LinearTransform)
        opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

        ivf_index = faiss.downcast_index(pq_index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)

        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.rev_swig_ptr(coarse_quantizer.get_xb(), ivf_index.pq.M * ivf_index.pq.dsub)
        coarse_embeds = coarse_embeds.reshape(-1)

        return pq_codes, centroid_embeds, coarse_embeds, opq_transform
