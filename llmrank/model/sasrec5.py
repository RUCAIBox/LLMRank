import torch
from recbole.model.sequential_recommender import SASRec as OriSASRec


class SASRec5(OriSASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def crop_seq(self, item_seq, item_seq_len):
        new_item_seq = torch.zeros_like(item_seq)
        new_item_seq_len = torch.zeros_like(item_seq_len)

        for i in range(item_seq.shape[0]):
            cur_len = item_seq_len[i].cpu().numpy().item()
            if cur_len > 5:
                new_item_seq_len[i] = 5
                cur_len = 5
            else:
                new_item_seq_len[i] = item_seq_len[i]
            new_item_seq[i, 0:cur_len] = item_seq[i, item_seq_len[i]-cur_len:item_seq_len[i]]
        return new_item_seq, new_item_seq_len

    def forward(self, item_seq, item_seq_len):
        new_item_seq, new_item_seq_len = self.crop_seq(item_seq, item_seq_len)
        return super().forward(new_item_seq, new_item_seq_len)

    def predict_on_subsets(self, interaction, idxs):
        candidate_item_emb = self.item_embedding(torch.LongTensor(idxs).to(self.device))
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)

        candidate_scores = (seq_output.unsqueeze(1) * candidate_item_emb).sum(dim=-1) # (B, C)
        candidate_scores = candidate_scores.cpu().numpy()

        scores = torch.full((item_seq.shape[0], self.n_items), -10000.)
        for i in range(idxs.shape[0]):
            for j in range(idxs.shape[1]):
                scores[i, idxs[i, j]] = float(candidate_scores[i, j])
        return scores
