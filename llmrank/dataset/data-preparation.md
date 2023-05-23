## Data Preparation Guideline

### Movielens 1M (ML-1M)

1. Download the raw atomic files from RecBole's dataset hub. [[link]](https://github.com/RUCAIBox/RecSysDatasets)
2. Sample 200 users and corresponding candidate items for experiments.
    ```bash
    # generate ml-1m.random
    cd llmrank/
    python sample_candidates.py -s random
    ```

    Note that `ml-1m.random` has the format `user_id<\t>candidate_item_1 candidate_item_2 ...`:
    ```
    5077	1116 884 1081 ...
    5075	3913 749 445 ...
    ```
    For each user in `ml-1m.random`, there are 100 randomly selected candidate items. We store these candidates for fair comparison between different variants.

    When we do experiments using `python evaluate.py -m Rank`, the number of items used to construct the candidate set depends on the hyperparameter `recall_budget`. For example, when `recall_budget=20`, the first 20 items of each line in `ml-1m.random` will be used.

    Then if `has_gt=True` (ground truth items are guaranteed to appear in the candidate set), the ground truth item will be appended into the candidate set (implemented in [`trainer.py`](../trainer.py), lines 70-90).

### Amazon Review - Games (Games)

*To be updated soon.*
