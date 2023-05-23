import random

from .rank import Rank


class RandOrder(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.rand_user_his = None

    def get_batch_inputs(self, interaction, idxs, i):
        user_his_text, candidate_text, candidate_text_order, candidate_idx = \
            super().get_batch_inputs(interaction, idxs, i)
        if self.rand_user_his is None:
            random.shuffle(user_his_text)
            self.rand_user_his = user_his_text
        return self.rand_user_his, candidate_text, candidate_text_order, candidate_idx
