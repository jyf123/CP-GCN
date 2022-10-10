# CP-GCN

This is the implementation of [Supporting Medical Relation Extraction via Causality-Pruned Semantic
Dependency Forest](https://arxiv.org/abs/2208.13472) at COLING 2022.
You can download the datasets and our pre-trained model in [here](https://drive.google.com/drive/folders/1m3wlxXcMLBIbYJ9foZfC8uA_dAMKoTPW?usp=sharing)
The code includes two datasets: CPR and PGR, both of them is avilable `./dataset`. The PubMed dataset is available in https://github.com/Cartus/AGGCN/tree/master/PubMed.


## Requirements


- Python 3 (tested on 3.8.8)

- PyTorch (tested on 1.10.1)

- CUDA (tested on 11.4)

- tqdm, networkx

- unzip, wget (for downloading only)


## Preparation

First, download and unzip GloVe vectors:

```
sh download.sh
```

Then prepare vocabulary and initial word vectors for different datasets (cpr/pgr). Take CPR as an example:

```
python prepare_vocab.py dataset/cpr dataset/cpr --glove_dir dataset/glove
```

  

This will write vocabulary and word vectors as a numpy matrix into their corresponding dir `./dataset/cpr`.

## Training a task-specific explainer


We have released the causal explanation dataset for our cpr and pgr dataset in `./distillation/dataset/`. To train the task-specific causal explainer, run:
```
sh training_explainer.sh
```
Model will be saved to `./explanation/cpr_top20`.

We have released the trained task-specific causal explainer in `./explanation/cpr_top20_old/`


## Training CP-GCN

To train the CP-GCN model, run:

```
sh train_cpr.sh
```

Model checkpoints and logs will be saved to `./saved_models/cpr`. 

## Evaluation for CP-GCN

Our pre-trained model is saved under the dir `./saved_models/cpr`. To run evaluation on the test set, run:

```
python eval.py saved_model/cpr
```

  
## Citation

```
@article{jin2022supporting,
  title={Supporting Medical Relation Extraction via Causality-Pruned Semantic Dependency Forest},
  author={Jin, Yifan and Li, Jiangmeng and Lian, Zheng and Jiao, Chengbo and Hu, Xiaohui},
  journal={arXiv preprint arXiv:2208.13472},
  year={2022}
}
```


