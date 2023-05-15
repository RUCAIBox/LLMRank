import os
import numpy as np
from tqdm import tqdm
import torch
from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color
from recbole.data.interaction import Interaction


class SelectedUserTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.selected_user_suffix = config['selected_user_suffix']  # candidate generation model, by default, random
        self.recall_budget = config['recall_budget']                # size of candidate Sets, by default, 20
        self.fix_pos = config['fix_pos']                            # whether fix the position of ground-truth items in the candidate set, by default, -1
        self.selected_uids, self.sampled_items = self.load_selected_users(config, dataset)

    def load_selected_users(self, config, dataset):
        selected_users = []
        sampled_items = []
        selected_user_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.selected_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        with open(selected_user_file, 'r', encoding='utf-8') as file:
            for line in file:
                uid, iid_list = line.strip().split('\t')
                selected_users.append(uid)
                sampled_items.append([item_token2id[_] if (_ in item_token2id) else 0 for _ in iid_list.split(' ')])
        selected_uids = list([user_token2id[_] for _ in selected_users])
        return selected_uids, sampled_items

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        new_inter = {
            col: torch.stack([inter[col] for inter in selected_interactions]) for col in selected_interactions[0].columns
        }
        selected_interactions = Interaction(new_inter)
        selected_pos_i = torch.stack(selected_pos_i)
        selected_pos_u = torch.arange(selected_pos_i.shape[0])

        if self.config['has_gt']:
            self.logger.info('Has ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            for i in range(idxs.shape[0]):
                if selected_pos_i[i] in idxs[i]:
                    pr = idxs[i].numpy().tolist().index(selected_pos_i[i].item())
                    idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])

            idxs = idxs[:,:self.recall_budget - 1]
            if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                idxs = torch.cat([idxs, selected_pos_i.unsqueeze(-1)], dim=-1).numpy()
            elif self.fix_pos == 0:
                idxs = torch.cat([selected_pos_i.unsqueeze(-1), idxs], dim=-1).numpy()
            else:
                idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                idxs = torch.cat([idxs_a, selected_pos_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
        else:
            self.logger.info('Does not have ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            idxs = idxs[:,:self.recall_budget]
            idxs = idxs.numpy()

        if self.fix_pos == -1:
            self.logger.info('Shuffle ground truth')
            for i in range(idxs.shape[0]):
                np.random.shuffle(idxs[i])
        scores = self.model.predict_on_subsets(selected_interactions.to(self.device), idxs)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        self.eval_collector.eval_batch_collect(
            scores, selected_interactions, selected_pos_u, selected_pos_i
        )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result
