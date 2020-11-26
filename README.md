# Adaptive Transformers for Learning Multimodal Representations

<h4>
ACL SRW 2020
</br>
Prajjwal Bhargava
</h4>
<hr>


**Paper:** [arXiv](https://arxiv.org/abs/2005.07486)


ML Code Completeness Checklist:
- [x] Specification of dependencies
- [x] Training code
- [x] Evaluation code
- [x] Pre-trained models
- [x] README file including table of results accompanied by precise commands to run/produce those results

## Dependencies:
Please refer `requirements.txt`.
To install,
```
$ pip install -r requirements.txt 
```

## Dataset Preparation
- Download the raw VQA 2.0 dataset from the [official website](https://visualqa.org/download.html).

Make sure that your data directory looks similar to the following structure (you can change the paths if you want a different structure in `train.py`).

- These instructions are from [LXMERT repo]((https://github.com/airsplay/lxmert#vqa)). Download the re-distributed JSON files.
```
mkdir -p data/vqa
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/train.json -P data/vqa/
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/nominival.json -P  data/vqa/
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/vqa/minival.json -P data/vqa/
```
For downloading FasterRCNN features, use these instructions:
```
mkdir -p data/mscoco_imgfeat
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/train2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/train2014_obj36.zip -d data/mscoco_imgfeat && rm data/mscoco_imgfeat/train2014_obj36.zip
wget --no-check-certificate https://nlp1.cs.unc.edu/data/lxmert_data/mscoco_imgfeat/val2014_obj36.zip -P data/mscoco_imgfeat
unzip data/mscoco_imgfeat/val2014_obj36.zip -d data && rm data/mscoco_imgfeat/val2014_obj36.zip
```
If the links don't work, you can use [Google drive link](https://drive.google.com/drive/folders/1Gq1uLUk6NdD0CcJOptXjxE6ssY5XAuat?usp=sharing) to get access. For more details, please refer [LXMERT repo](https://github.com/airsplay/lxmert).

Setup the directory structure like this:
In `/home/user/`
```
+-- data
|   +-- lxmert
|   +-- mscoco_imgfeat
|   +-- vqa
+-- adaptive_transformer
+-- snap
.......
```
Create a directory snap, that's where checkpoints will be store by default.
All of this structure can be changed but suitable modifications will be needed in `train.py`.

FasterRCNN features are loaded all at once in the RAM, so you'd require an instance with >48 GB of RAM. For training, I used a single P100 Nvidia GPU. 

## Downloading pretrained model
Please download the pretrained models from this [Google drive link](https://drive.google.com/drive/folders/1V1SjSfGCqBJZi2INzCmNKxnoXI4FnVeP?usp=sharing)

Alternatively, if you want to train (finetune) the model yourself, download the pretrained weights from [here](http://nlp1.cs.unc.edu/data/model_LXRT.pth). Skip this step if you're using my weights.

## Training
```
$ git clone https://github.com/prajjwal1/adaptive_transformer
$ cd adaptive_transformer
$ python3 train.py --bs=128 --epochs=1 --sparse --tiny #test script
```
If this worked well, then you're ready to train.

Usage:
```
python train.py
    [--bs]            # Specify the batch size
    [--epochs]        # Specify the epochs
    [--tiny]          # Runs a test example (for debugging purposes)  
    [--adaptive]      # Uses Adaptive Attention Span
    [--sparse]        # Uses Entmax from Adaptively Sparse Transformers instead of softmax
    [--layerdrop]     # Enables layerdrop
    [--load_model]    # Resume training by specifying a checkpoint
    [--test]          # Dumps a JSON file for submission to VQA servers.
```
More customizations can be done by modifying the `params` and `config` dict in `train.py`. 

It looks like this 
```
params = {
    "adapt_span_enabled": args.adaptive,
    "attn_span": 1024,
    "adapt_span_loss_coeff": 0.000005,
    "adapt_span_ramp": 32,
    "adapt_span_init": 0.002,
    "adapt_span_cache": True,
    "nb_heads": 12,
    "bs": args.bs,
    "mask_size": [20, 36],
    "sparse_enabled": args.sparse,
    "num_attention_heads": 4,
    "layer_sizes": {"lang": 9, "cross": 5, "vision": 5},
    "from_scratch": False,
    "layerdrop_enabled": args.layerdrop,
    "layerdrop_num_layers": 1,
}
config = {
    "adaptive_enable": args.adaptive,
    "sparse_enable": args.sparse,
    "measure_flops": False,
    "load_model": args.load_model,
}
```
Please check the `params` dict when starting training to see the configurations. Config should match with the config used in loaded model. 
Remove the `tiny` flag to train on whole dataset.

##  Using Adaptive Attention Span
```
python train.py --bs=128 --epochs=1 --adaptive --tiny
```
 By default, attention spans of each layer is printed so that you can track it.

## Using Entmax 
If `sparse` flag is enabled, softmax will be replaced with entmax to compute probability distribution of attention weights.
```
python train.py --bs=128 --epochs=1 --sparse --tiny
```

## Using Layerdrop
```
python train.py --bs=128 --epochs=1 --layerdrop --tiny
```
Specify the following as per use case in `train.py`:
- `params['layerdrop_num_layers']`  # Number of layers to drop
- `params['layer_sizes']` # Number of layers you require

NOTE: Number of layers `params['layer_sizes']` have to match with number of layers in the model checkpoint. To perform pruning during inference, default `learn.load` method is not suitable as it loads all the layers. Please refer to this [fairseq issue](https://github.com/pytorch/fairseq/issues/1667#issuecomment-581595354) to perform pruning during inference.

### Resuming Training

To load a model trained with `adaptive` or `sparse` or `layerdrop` flag:
```
python train.py --bs=128 --epochs=1 --adaptive --tiny --load_model=adaptive_6910
python train.py --bs=128 --epochs=1 --sparse --tiny --load_model=sparse_7
python train.py --bs=128 --epochs=1 --layerdrop --load_model=layerdrop_1066_ldrop_1 --tiny
```

### Running evaluation on test data set
```
python train.py --bs=128 --test --adaptive --load_model=adaptive_6910
```
When `test` flag is passed, only inference is performed on the test set. Ground truths for test set for VQA are not publicly available. This command will dump the JSON file in the `/snap` directory. Submit the JSON file through the [EvalAI competition page](https://evalai.cloudcv.org/web/challenges/challenge-page/514/overview).

## Explanation of this codebase
- `dataset` : contains standard Pytorch dataset class for VQA
- `models`: Contains implmentation of adaptive mechanisms and LXMERT
- `nbs`: Probably the most interesting part. Use this to understand my workflow, attention methods I used. I used these notebooks to develop this codebase. You can also use these to understand how attention works in this context and much more.
-  `optimizers`: implementation of LAMB and Lookahead optimizer
- `pretrain`: utility tools
- `train.py`: Specifies how training and testing to be carried out. You'd probably want to modify this to adapt to your work.
- `learner.py`: Implements a Learner class to control all functionalities of this codebase.
- `run_train.sh`: You can modify this to setting hardware specific training (Optional)
- `run_test.sh`: Set of tests (for me).

## Inference: Visualizing results
Please refer to [nbs/inference.ipynb](https://github.com/prajjwal1/adaptive_transformer/blob/master/nbs/inference.ipynb) to load your trained model, obtain predictions and visualize the results.

## Results

These results can be reproduced by using the scripts I provided above and using the same `params` and `config` dict values.
Our model achives the following performance on the VQA 2.0 benchmark:
```
| Model                                 | test-dev | test-std |
|---------------------------------------|----------|----------|
| LXMERT                                |          |          |
| w/ softmax                            | 72.42    | 72.54    |
| w/ Adaptive Attention Span            | 71.62    | 71.72    |
| w/ Adaptive Sparse                    | 71.73    | 71.97    |
| w/ Layerdrop (10-6-6, p=1)            | 66.4     | 66.72    |
| w/ Layerdrop (10-6-6, p=0)            | 66.35    | 66.57    |
| w/ Layerdrop (9-5-5, p=1)             | 66.51    | 66.81    |
| w/ Adaptive Attention Span and Entmax | 63.07    | 63.33    |
```

## Citation
If you use this work in any form, please cite the paper:
```
@inproceedings{bhargava-2020-adaptive,
    title = "Adaptive Transformers for Learning Multimodal Representations",
    author = "Bhargava, Prajjwal",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-srw.1",
    doi = "10.18653/v1/2020.acl-srw.1",
    pages = "1--7",
    abstract = "The usage of transformers has grown from learning about language semantics to forming meaningful visiolinguistic representations. These architectures are often over-parametrized, requiring large amounts of computation. In this work, we extend adaptive approaches to learn more about model interpretability and computational efficiency. Specifically, we study attention spans, sparse, and structured dropout methods to help understand how their attention mechanism extends for vision and language tasks. We further show that these approaches can help us learn more about how the network perceives the complexity of input sequences, sparsity preferences for different modalities, and other related phenomena.",
}
```

## Acknowledgement
- Code for LXMERT Model was adapted from [LXMERT](https://github.com/airsplay/lxmert) repo.
- Entmax autograd function implementation was adapted from [entmax repo](https://github.com/deep-spin/entmax)

