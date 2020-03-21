# Adaptive Transformer
The following is the code for the paper

Usage:
`python fit.py
    [--bs]
    [--epochs]
    [--tiny]
    [--adaptive]
    [--sparse]
    [--layerdrop]
    [--load_model]
    [--test]
`
## Data
Setup the directory structure like this:
In `/home/user/`
```
+-- Snap
|   +-- lxmert
|   +-- mscoco_imgfeat
|   +-- vqa
+-- lxmert_adaptive
+-- models
.......
```
This structure can be changed but suitable modifications will be needed in `fit.py`
##  Using Adaptive Attention Span
```
python fit.py --bs=128 --epochs=1 --adaptive --tiny
```
Remove the `tiny` flag to train on whole dataset. By default, attention spans of each layer is printed so that you can track it.

## Using Entmax 
If `sparse` flag is enabled, softmax will be replaced with entmax to compute probability distribution of attention weights.
```
python fit.py --bs=128 --epochs=1 --sparse --tiny
```

## Using Layerdrop
```
python fit.py --bs=128 --epochs=1 --layerdrop --tiny
```
Specify the following as per use case in `fit.py`:
- `params['layerdrop_num_layers']`  # Number of layers to drop
- `params['layer_sizes']` # Number of layers you require

NOTE: Number of layers `params['layer_sizes']` have to match with number of layers in the model checkpoint. To perform pruning during inference, default `learn.load` method is not suitable as it loads all the layers. Please refer to this [fairseq issue](https://github.com/pytorch/fairseq/issues/1667#issuecomment-581595354) to perform pruning during inference.

### Resuming Training

To load a model trained with `adaptive` or `sparse` or `layerdrop` flag:
```
python fit.py --bs=128 --epochs=1 --adaptive --tiny --load_model=adaptive_6910
python fit.py --bs=128 --epochs=1 --sparse --tiny --load_model=sparse_7
python fit.py --bs=128 --epochs=1 --layerdrop --load_model=layerdrop_1066_ldrop_1 --tiny
```

### Running evaluation on test data set
```
python fit.py --bs=128 --test --adaptive --load_model=adaptive_6910
```
When `test` flag is passed, only inference is performed on the test set. Ground truths for test set for VQA are not publicly available. This command will dump the JSON file in the `/snap` directory. Submit the JSON file in the [EvalAI competition page](https://evalai.cloudcv.org/web/challenges/challenge-page/514/overview).

## Acknowledgement
- Code for LXMERT was adapted from [LXMERT](https://github.com/airsplay/lxmert) repo.
- Entmax autograd function was adapted from [entmax repo](https://github.com/deep-spin/entmax)
