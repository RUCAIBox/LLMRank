r"""
Pop
################################################

"""

import torch

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class Pop(SequentialRecommender):
    r"""Pop is an fundamental model that always recommend the most popular item."""
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Pop, self).__init__(config, dataset)

        self.item_cnt = torch.zeros(
            self.n_items, 1, dtype=torch.long, device=self.device, requires_grad=False
        )
        self.max_cnt = None
        self.fake_loss = torch.nn.Parameter(torch.zeros(1))
        self.other_parameter_name = ["item_cnt", "max_cnt"]

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        item = interaction[self.ITEM_ID]
        seq_len = interaction[self.ITEM_SEQ_LEN]
        add_item = interaction[self.ITEM_SEQ][:, 0]
        pad_id = torch.zeros_like(item)
        self.item_cnt[item, :] = self.item_cnt[item, :] + 1
        add_item = torch.where(seq_len == 1, add_item, pad_id)
        self.item_cnt[add_item, :] = self.item_cnt[add_item, :] + 1
        self.max_cnt = torch.max(self.item_cnt, dim=0)[0]

        return torch.nn.Parameter(torch.zeros(1))

    def full_sort_predict(self, interaction):
        batch_user_num = interaction[self.USER_ID].shape[0]
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = torch.repeat_interleave(result.unsqueeze(0), batch_user_num, dim=0)
        return result.view(-1)

    def predict_on_subsets(self, interaction, idxs):
        item_seq = interaction[self.ITEM_SEQ]
        result = self.item_cnt.to(torch.float64) / self.max_cnt.to(torch.float64)
        result = result.squeeze(-1)
        scores = torch.full((item_seq.shape[0], self.n_items), -10000.)
        for i in range(idxs.shape[0]):
            for j in range(idxs.shape[1]):
                scores[i, idxs[i, j]] = result[idxs[i, j]]
        return scores
