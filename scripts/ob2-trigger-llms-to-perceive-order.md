## Observation 2. LLMs can be triggered to perceive the orders

By employing specifically designed promptings, such as recency-focused prompting and in-context learning, *LLMs can be triggered to perceive the order* of historical user behaviors, leading to improved ranking performance.

<div align="center"> 
<img src='../assets/tab-2.png' width="75%">
<div>Table 2</div>
</div>

### Zero-Shot Methods

**Ours**

- Sequential [[code]](../llmrank/model/rank.py)

    ```bash
    cd llmrank/

    # ML-1M
    python evaluate.py -m Rank

    # Games
    python evaluate.py -m Rank -d Games
    ```

- Recency-Focused [[code]](../llmrank/model/rf.py)

    ```bash
    cd llmrank/

    # ML-1M
    python evaluate.py -m RF

    # Games
    python evaluate.py -m RF -d Games
    ```

- In-Context Learning [[code]](../llmrank/model/icl.py)

    ```bash
    cd llmrank/

    # ML-1M
    python evaluate.py -m ICL

    # Games
    python evaluate.py -m ICL -d Games
    ```

**Baselines**

- BM25

    ```bash
    cd llmrank/

    # ML-1M
    python evaluate.py -m BM25

    # Games
    python evaluate.py -m BM25 -d Games
    ```

- UniSRec

    Download `*.feat1CLS` and `*_item_dataset2row.npy` from [[link]](https://drive.google.com/drive/folders/16hdqUCNOj9M1dApWYN0iGND_0WoMRyGh?usp=share_link), and download `UniSRec-FHCKM-300.pth` from [[link]](https://drive.google.com/drive/folders/17Em-qAhZ8ybcBah3EdmAcQWfn1D8ONh-?usp=sharing).

    ```bash
    cd llmrank/

    # ML-1M
    python evaluate.py -m UniSRec -p pretrained_models/UniSRec-FHCKM-300.pth

    # Games
    python evaluate.py -m UniSRec -d Games -p pretrained_models/UniSRec-FHCKM-300.pth
    ```

### Conventional Methods

> Pre-trained models can be downloaded following the instructions in [[downloading pre-trained models]](../llmrank/pretrained_models/README.md).

- BPRMF

    ```bash
    cd llmrank/

    # ML-1M
    # python run_baseline.py -m BPR -d ml-1m
    # mv xxx.pth pretrained_models/BPR-ml-1m.pth
    python evaluate.py -m BPR -p pretrained_models/BPR-ml-1m.pth

    # Games
    # python run_baseline.py -m BPR -d Games
    # mv xxx.pth pretrained_models/BPR-Games.pth
    python evaluate.py -m BPR -d Games -p pretrained_models/BPR-Games.pth
    ```

- Pop

    ```bash
    cd llmrank/

    # ML-1M
    # python run_baseline.py -m Pop -d ml-1m
    # mv xxx.pth pretrained_models/Pop-ml-1m.pth
    python evaluate.py -m Pop -p pretrained_models/Pop-ml-1m.pth

    # Games
    # python run_baseline.py -m Pop -d Games
    # mv xxx.pth pretrained_models/Pop-Games.pth
    python evaluate.py -m Pop -d Games -p pretrained_models/Pop-Games.pth
    ```

- SASRec

    ```bash
    cd llmrank/

    # ML-1M
    # python run_baseline.py -m SASRec -d ml-1m
    # mv xxx.pth pretrained_models/SASRec-ml-1m.pth
    python evaluate.py -m SASRec -p pretrained_models/SASRec-ml-1m.pth

    # Games
    # python run_baseline.py -m SASRec -d Games
    # mv xxx.pth pretrained_models/SASRec-Games.pth
    python evaluate.py -m SASRec -d Games -p pretrained_models/SASRec-Games.pth
    ```
