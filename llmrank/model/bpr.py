r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class BPR(SequentialRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE
    
    def __init__(self, config, dataset):
        super(BPR, self).__init__(config, dataset)
        self.n_users = dataset.num(self.USER_ID)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]
        seq_len = interaction[self.ITEM_SEQ_LEN]
        add_item = interaction[self.ITEM_SEQ][:, 0]

        pad_id = torch.zeros_like(user)
        add_user = torch.where(seq_len == 1, user, pad_id)
        add_item = torch.where(seq_len == 1, add_item, pad_id)

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(user_e, neg_e).sum(dim=1)
        loss = self.loss(pos_item_score, neg_item_score)

        add_user_e, add_pos_e = self.forward(add_user, add_item)
        add_pos_item_score, add_neg_item_score = torch.mul(add_user_e, add_pos_e).sum(dim=1), torch.mul(add_user_e, neg_e).sum(dim=1)

        batch_add_loss = -torch.log(1e-10 + torch.sigmoid(add_pos_item_score - add_neg_item_score))
        add_loss = (batch_add_loss * (seq_len == 1)).mean()
        return loss + add_loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

    def predict_on_subsets(self, interaction, idxs):
        candidate_item_emb = self.item_embedding(torch.LongTensor(idxs).to(self.device))

        user = interaction[self.USER_ID]
        user_emb = self.user_embedding(user)

        candidate_scores = (user_emb.unsqueeze(1) * candidate_item_emb).sum(dim=-1)
        candidate_scores = candidate_scores.cpu().numpy()

        scores = torch.full((user.shape[0], self.n_items), -10000.)
        for i in range(idxs.shape[0]):
            for j in range(idxs.shape[1]):
                scores[i, idxs[i, j]] = float(candidate_scores[i, j])

        return scores
