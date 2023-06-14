# MarkupMnA: Markup-Based Segmentation of M&A Agreements

[MarkupMnA: Markup-Based Segmentation of M&A Agreements]\
Sukrit Rao, Pranab Islam, Rohith Bollineni, Shaan Khosla, Tingyi Fei, Qian Wu, Kyunghyun Cho, Vladimir A Kobzar
<!-- [![DOI]() -->

**This repo is no longer being maintained. For the code related to the paper 
MarkupMnA: Markup-Based Segmentation of M&A Agreements please visit [[the new repo](https://github.com/MarkupMnA/MarkupMnA-Markup-Based-Segmentation-of-MnA-Agreements)]**

This repo contains:
- The **MarkupMnA: Markup-Based Segmentation of M&A Agreements Dataset** (through download script)
- **Training** and **evaluation** code for the experiments mentioned in MarkupMnA: Markup-Based Segmentation of M&A Agreements.

## Download the Dataset
To download and unpack the MarkupMnA dataset, use:
```
python download.py --savedir /path/to/savedir
```


This creates the following file structure:
```
{savedir}/contracts   # contains MarkupMnA base data (.csv files)
```


## MarkupMnA Dataset
The MarkupMnA Dataset can be found at [[zenodo](https://doi.org/10.5281/zenodo.8034853)]


To download the MarkupMnA dataset, use:
```
python download.py --savedir /path/to/savedir
```
## Training

The provided code supports:
- Training the **MarkupLM base** model
- Training the **MarkupLM large** model
- Performing the **ablation experiments** mentioned in MarkupMnA: Markup-Based Segmentation of M&A Agreements
- Training on a **subset of k** contracts from the training dataset

#### Training the MarkupLM base model
```bash
python3 train.py --config=./configs/config.yaml
```

- There are different config files present in the `configs` directory. You can 
execute a job from the list above by providing the path to the config file for that 
particular job
- The output created by the above command will be saved in the `collateral dir` 
    specified in the config file. 
 

## Evaluation
To obtain performance metrics on the test set you can run 
```bash
python3 inference.py --config=./configs/config.yaml
```

**Note: Make sure to update the `predict` key in the config file before running 
this command**

We also provide a script to obtain performance metrics using
constrained decoding mentioned in MarkupMnA: Markup-Based Segmentation of M&A Agreements. You can run it using 
```bash
python3 constrained_inference.py --config=./configs/config.yaml
```