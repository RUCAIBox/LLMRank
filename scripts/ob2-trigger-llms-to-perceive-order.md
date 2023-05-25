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

### Conventional Methods

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
