import torch

from .rank import Rank


class FakeHis(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.fake_user_his = None

    def get_batch_inputs(self, interaction, idxs, i):
        if self.fake_user_his is None:
            user_his = interaction[self.ITEM_SEQ]
            tmp_user_his = torch.randint_like(user_his, 1, self.n_items)
            self.fake_user_his = torch.where(user_his > 0, tmp_user_his, user_his)
        user_his_len = interaction[self.ITEM_SEQ_LEN]

        real_his_len = min(self.max_his_len, user_his_len[i].item())
        user_his_text = [str(j) + '. ' + self.item_text[self.fake_user_his[i, user_his_len[i].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]
        candidate_text = [self.item_text[idxs[i,j]]
                for j in range(idxs.shape[1])]
        candidate_text_order = [str(j) + '. ' + self.item_text[idxs[i,j].item()]
                for j in range(idxs.shape[1])]
        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx
