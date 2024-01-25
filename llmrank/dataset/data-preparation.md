## Data Preparation Guideline

### Movielens 1M (ML-1M)

1. Download the raw atomic files from RecBole's dataset hub. [[link]](https://github.com/RUCAIBox/RecSysDatasets)
2. Sample 6,040 (all) users and corresponding candidate items for experiments.
    ```bash
    # generate ml-1m-full.random
    cd llmrank/
    python sample_candidates.py -d ml-1m-full -u 6040 -s random
    ```

    Note that `ml-1m-full.random` has the format `user_id<\t>candidate_item_1 candidate_item_2 ...`:
    ```
    5077	1116 884 1081 ...
    5075	3913 749 445 ...
    ```
    For each user in `ml-1m-full.random`, there are 100 randomly selected candidate items. We store these candidates for fair comparison between different variants.

    When we do experiments using `python evaluate.py -m Rank -d ml-1m-full`, the number of items used to construct the candidate set depends on the hyperparameter `recall_budget`. For example, when `recall_budget=20`, the first 20 items of each line in `ml-1m-full.random` will be used.

    Then if `has_gt=True` (ground truth items are guaranteed to appear in the candidate set), the ground truth item will be appended into the candidate set (implemented in [`trainer.py`](../trainer.py), lines 70-90).
3. For different candidate retrieval strategies, please replace `-s random` to `-s bm25`, `-s bert`, `-s pop`, `-s bpr`, `-s gru4rec`, or `-s sasrec` in step 2.
4. For generating candidates which are retrieved by multiple strategies (`.rabmbepobpgrsa_3`, denotes for concatenating the first two letters of each strategy name).
    ```bash
    python mix_sources.py
    ```

### Amazon Review - Games (Games)

1. Download raw datasets from Amazon review data 2018, including the metadata and ratings only data of each category. [[link]](https://nijianmo.github.io/amazon/index.html)
Here is an example.
    ```
    dataset/
      raw/
        Metadata/
          meta_Video_Games.json.gz
        Ratings/
          Video_Games.csv
    ```
2. Process downstream datasets.
    ```
    cd llmrank/
    python data_process_amazon.py --dataset Games
    ```
    Note that following [[UniSRec]](https://github.com/RUCAIBox/UniSRec), we split the data into separate files for training, validation and evaluation: `.train.inter`, `.valid.inter`, `.test.inter` .

3. Repeat the second step in processing ML-1M datasets, for example:
    ```
    # generate Games-6k.random
    cd llmrank/
    python sample_candidates.py -d Games-6k -u 6000 -s random
    ```

## Auxiliary Files of UniSRec / VQ-Rec Preparation Guideline

### UniSRec

1. Generate item embedding from item related text (title).
    ```
    cd llmrank/
    python dataset/unisrec_auxiliary_files_process.py
    ```

### VQ-Rec

1. Run step 1 of UniSRec to obtain `*.feat1CLS`.
2. Generate the item index file.
    ```
    python dataset/vqrec_auxiliary_files_process.py --dataset ml-1m
    ```
