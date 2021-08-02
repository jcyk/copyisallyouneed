# Neural Machine Translation with Monolingual Translation Memory

Code for our ACL2021 paper
[Neural Machine Translation with Monolingual Translation Memory](https://arxiv.org/pdf/2105.11269.pdf)

## Data

The preprocessed JRC data is available at [Google Drive](https://drive.google.com/file/d/1iuBH_YsnL28cTYjjpSq5BgukG7QhBLs_/view?usp=sharing).

## Environment 

The code is written and tested with the following packages:

- transformers==2.11.0
- faiss-gpu==1.6.1
- torch==1.5.1+cu101

## Instructions

The scripts to reproduce our results can be found in the `scripts` folder. Here we give an example to reproduce our experiments (es=>en translation). NOTE: You should check detailed information in the corresponding shell scripts.

0. do `export MTPATH=where_you_hold_your_data_and_models`
1. data preprocessing: `sh scripts/prepare.sh` 
2. cross-alignment pre-training for the retrieval model: `sh scripts/esen/pretrain.sh`

3. build the initial index: `sh scripts/esen/build_index.sh`
4. training: `sh scripts/esen/train.multihead.dynamic.sh` (model #4: fixed $E_{tgt}$) or `sh scripts/esen/train.multihead.dynamic.sh` (model #5)
5. testing:   `sh scripts/work.sh` (model #4)  or `sh scripts/work1.sh` (model #5)

Other baselines:

For model #1, see `sh scripts/train.vanilla.sh` .

For model #2, see `sh scripts/train.bm25.sh`.

For model #3, see `sh scripts/train.static.sh`

## Citation

```
@inproceedings{cai-etal-2021-neural,
    title = "Neural Machine Translation with Monolingual Translation Memory",
    author = "Cai, Deng  and
      Wang, Yan  and
      Li, Huayang  and
      Lam, Wai  and
      Liu, Lemao",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.567",
    doi = "10.18653/v1/2021.acl-long.567",
    pages = "7307--7318",
    abstract = "Prior work has proved that Translation Memory (TM) can boost the performance of Neural Machine Translation (NMT). In contrast to existing work that uses bilingual corpus as TM and employs source-side similarity search for memory retrieval, we propose a new framework that uses monolingual memory and performs learnable memory retrieval in a cross-lingual manner. Our framework has unique advantages. First, the cross-lingual memory retriever allows abundant monolingual data to be TM. Second, the memory retriever and NMT model can be jointly optimized for the ultimate translation goal. Experiments show that the proposed method obtains substantial improvements. Remarkably, it even outperforms strong TM-augmented NMT baselines using bilingual TM. Owning to the ability to leverage monolingual data, our model also demonstrates effectiveness in low-resource and domain adaptation scenarios.",
}
```

## Contact

[Deng Cai](https://jcyk.github.io/)
