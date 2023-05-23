import torch
from recbole.model.sequential_recommender import SASRec as OriSASRec


class SASRec(OriSASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

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
