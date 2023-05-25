# LLMRank

**LLMRank** aims to investigate the capacity of LLMs that act as the ranking model for recommender systems.

See our paper: [Large Language Models are Zero-Shot Rankers for Recommender Systems](https://arxiv.org/abs/2305.08845)

## 🛍️ LLMs as Zero-Shot Rankers

![](assets/model.png)

We use LLMs as ranking models in an instruction-following paradigm. For each user, we first construct two natural language patterns that contain **sequential interaction histories** and **retrieved candidate items**, respectively. Then these patterns are filled into a natural language template as the final instruction. In this way, LLMs are expected to understand the instructions and output the ranking results as the instruction suggests.

## 🚀 Quick Start

1. Write your own OpenAI API keys into [`llmrank/openai_api.yaml`](https://github.com/RUCAIBox/LLMRank/blob/master/llmrank/openai_api.yaml).
2. Unzip dataset files.
    ```bash
    cd llmrank/dataset/ml-1m/; unzip ml-1m.inter.zip
    cd llmrank/dataset/Games/; unzip Games.inter.zip
    ```
    For data preparation details, please refer to [[data-preparation]](llmrank/dataset/data-preparation.md).
3. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Evaluate ChatGPT's zero-shot ranking abilities on ML-1M dataset.
    ```bash
    cd llmrank/
    python evaluate.py -m Rank
    ```

## 🔍 Key Findings

> Please click the links below each "Observation" to find the code and scripts to reproduce the results.

### Observation 1. LLMs struggle to perceive order of user histories

LLMs can utilize historical behaviors for personalized ranking, but *struggle to perceive the order* of the given sequential interaction histories.

[[reproduction scripts]](scripts/ob1-struggle-to-perceive-order.md)

### Observation 2. LLMs can be triggered to perceive the orders

By employing specifically designed promptings, such as recency-focused prompting and in-context learning, *LLMs can be triggered to perceive the order* of historical user behaviors, leading to improved ranking performance.

[[reproduction scripts & benchmark 1 (Table 2)]](scripts/ob2-trigger-llms-to-perceive-order.md)

### Observation 3. Promising zero-shot ranking abilities

LLMs have promising zero-shot ranking abilities, especially on candidates retrieved by multiple candidate generation models with different practical strategies.

[reproduction scripts (coming soon)]

[benchmark 2 (Table 3) (coming soon)]

### Observation 4. Biases exist in using LLMs to rank

LLMs suffer from position bias and popularity bias while ranking, which can be alleviated by specially designed prompting or bootstrapping strategies.

[reproduction scripts (coming soon)]

## 🌟 Acknowledgement

Please cite the following paper if you find our code helpful.

```bibtex
@article{hou2023llmrank,
  title={Large Language Models are Zero-Shot Rankers for Recommender Systems},
  author={Yupeng Hou and Junjie Zhang and Zihan Lin and Hongyu Lu and Ruobing Xie and Julian McAuley and Wayne Xin Zhao},
  journal={arXiv preprint arXiv:2305.08845},
  year={2023}
}
```

The experiments are conducted using the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

We use the released pre-trained models of [UniSRec](https://github.com/RUCAIBox/UniSRec) and [VQ-Rec](https://github.com/RUCAIBox/VQ-Rec) in our zero-shot recommendation benchmarks.

Thanks [@neubig](https://github.com/neubig) for the amazing implementation of asynchronous dispatching OpenAI APIs. [[code]](https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a)
