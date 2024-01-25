from .rank import Rank


class NoHis(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def construct_prompt(self, dataset_name, user_his_text, candidate_text_order):
        if dataset_name in ['ml-1m', 'ml-1m-full']:
            prompt = f"I'm a movie enthusiast and would like to watch some movies.\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most. Please think step by step.\n" \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif dataset_name in ['Games', 'Games-6k']:
                    f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                    f"Please only output the order numbers after ranking. Split these order numbers with line break."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
