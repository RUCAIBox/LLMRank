from .rank import Rank


class RF(Rank):
    """Recency-Focused Prompting
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    def construct_prompt(self, dataset_name, user_his_text, candidate_text_order):
        recent_item = user_his_text[-1][user_his_text[-1].find('. ') + 2:]
        if dataset_name in ['ml-1m', 'ml-1m-full']:
            prompt = f"I've watched the following movies in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate movies that I can watch next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} movies by measuring the possibilities that I would like to watch next most, according to my watching history. Please think step by step.\n" \
                    f"Note that my most recently watched movie is {recent_item}. " \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
        elif dataset_name in ['Games', 'Games-6k']:
            prompt = f"I've purchased the following products in the past in order:\n{user_his_text}\n\n" \
                    f"Now there are {self.recall_budget} candidate products that I can consider to purchase next:\n{candidate_text_order}\n" \
                    f"Please rank these {self.recall_budget} products by measuring the possibilities that I would like to purchase next most, according to the given purchasing records. Please think step by step.\n" \
                    f"Note that my most recently purchased product is {recent_item}. " \
                    f"Please show me your ranking results with order numbers. Split your output with line break. You MUST rank the given candidate movies. You can not generate movies that are not in the given candidate list."
                    # f"Please only output the order numbers after ranking. Split these order numbers with line break."
        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt
