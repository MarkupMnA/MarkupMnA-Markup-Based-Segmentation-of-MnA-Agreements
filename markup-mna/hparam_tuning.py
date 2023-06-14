import os
import yaml
import json
import argparse
import pandas as pd

from tqdm.auto import tqdm
from IPython.display import display

import torch
import evaluate
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from transformers import MarkupLMProcessor
from transformers import MarkupLMForTokenClassification
from transformers import set_seed

import optuna
from ray import tune
from ray.tune.search.optuna import OptunaSearch

import utils
import input_pipeline


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

# set the seed for the hugggingface package
set_seed(42)


def run_train_loop(batch, model, optimizer, scheduler, loss_fct,
                   device, train_metric, label_list, usr_cfg):
    '''Runs train loop for one batch of input

    Args:
        batch: dict[torch.Tensor]. Dict containing input tensors
            required for forward pass of MarkupLM
        model: transformers.PreTrainedModel. fine-tuned MarkupLM model
        optimizer: torch.optim.AdamW Optimizer used for training
        scheduler: torch.optim.lr_scheduler.StepLR Learning rate scheduler
        loss_fct: torch.nn.CrossEntropyLoss. Loss function used for training
        device: torch.device. Specifies whether GPU is available for computation
        train_metric: evaluate.seqeval. The metric used for
            computing the f1-score
        label_list: list. List of labels used to train the MarkupLM model
        config: dict. Contains user-provided params and args

    Returns:
        None
    '''
    # get the inputs
    inputs = {k: v.to(device) for k, v in batch.items()}

    # if ablation mode is set to true then
    # either mask the xpaths or shuffle them
    if user_cfg["ablation"]["run_ablation"]:
        inputs = utils.ablation(user_cfg, inputs)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = model(**inputs)

    # get the logits, labels
    logits = outputs.logits
    labels = inputs["labels"]

    # get the attention mask to determine valid tokens
    attention_mask = inputs["attention_mask"]

    # Only keep active parts of the loss
    num_labels = len(label_list)
    active_loss = attention_mask.view(-1) == 1
    active_logits = logits.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        labels.view(-1),
        torch.tensor(loss_fct.ignore_index).type_as(labels),
    )
    loss = loss_fct(active_logits, active_labels)

    loss.backward()
    optimizer.step()
    scheduler.step()

    print("Train Loss:", loss.item())

    predictions = outputs.logits.argmax(dim=-1)
    # labels = batch["labels"]
    preds, refs = utils.convert_preds_to_labels(predictions,
                                                labels,
                                                label_list,
                                                device)

    train_metric.add_batch(
        predictions=preds,
        references=refs,
    )

    return


def run_eval_loop(dataloader, model, device,
                  metric, label_list, user_cfg):
    '''Runs eval loop for entire dataset

    Args:
        dataloader: torch.utils.data.DataLoader: iterator over Dataset object
        model: transformers.PreTrainedModel. fine-tuned MarkupLM model
        device: torch.device. Specifies whether GPU is available for computation
        metric: evaluate.seqeval. The metric used for computing the f1-score
        label_list: list. List of labels used to train the MarkupLM model
        config: dict. Contains user-provided params and args

    Returns:
        None
    '''
    model.eval()
    for batch in tqdm(dataloader, desc='eval_loop'):
        # get the inputs;
        inputs = {k: v.to(device) for k, v in batch.items()}

        # if ablation mode is set to true then
        # either mask the xpaths or shuffle them
        if user_cfg["ablation"]["run_ablation"]:
            inputs = utils.ablation(user_cfg, inputs)

        # forward + backward + optimize
        outputs = model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        preds, refs = utils.convert_preds_to_labels(predictions,
                                                    labels,
                                                    label_list,
                                                    device)

        metric.add_batch(predictions=preds, references=refs)

    return


def objective(config):
    '''Main execution of script'''

    user_cfg = config['user_cfg']

    print("*" * 50)
    print("user_cfg", user_cfg)
    print("user_cfg data", user_cfg["data"])
    print("CWD", os.getcwd())
    print("*" * 50)

    # get the  list of labels along with the label to id mapping and
    # reverse mapping
    label_list, id2label, label2id = utils.get_label_list(user_cfg)

    print("*" * 50)
    print('Prepared Label List. Preparing Training Data ')
    print("*" * 50)

    # preprocess the train and eval dataset
    train_data = utils.get_dataset(
        user_cfg["data"]["train_contract_dir"],
        id2label,
        label2id,
        data_split='train',
        num_contracts=user_cfg['data']['data_split']
    )

    print("*" * 50)
    print('Prepared Training Data. Preparing Eval Data ')
    print("*" * 50)

    eval_data = utils.get_dataset(
        user_cfg["data"]["eval_contract_dir"],
        id2label,
        label2id,
        data_split='eval',
        num_contracts=None
    )

    print("*" * 50)
    print(f'Using Large Model: {user_cfg["model"]["use_large_model"]}')
    print(f"Label only first subword: {user_cfg['model']['label_only_first_subword']}")
    print("*" * 50)

    # define the processor and model
    if user_cfg["model"]["use_large_model"]:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-large",
            only_label_first_subword=user_cfg['model']['label_only_first_subword']
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-large", id2label=id2label, label2id=label2id
        )

    else:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-base",
            only_label_first_subword=user_cfg['model']['label_only_first_subword']
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-base", id2label=id2label, label2id=label2id
        )

    processor.parse_html = False

    # convert the input dataset
    # to torch datasets. Create the dataloaders as well
    train_dataset = input_pipeline.MarkupLMDataset(
        data=train_data,
        processor=processor,
        max_length=user_cfg["model"]["max_length"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=user_cfg["model"]["train_batch_size"],
        shuffle=True
    )

    eval_dataset = input_pipeline.MarkupLMDataset(
        data=eval_data,
        processor=processor,
        max_length=user_cfg["model"]["max_length"]
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=user_cfg["model"]["eval_batch_size"],
        shuffle=False
    )

    # get the class weights used to weigh the different terms in the loss fn
    class_value_counts, class_weights = utils.get_class_dist(user_cfg["data"]["train_contract_dir"],
                                                             id2label,
                                                             label2id,
                                                             mode='inverse')

    # Define hyperparameters to tune
    num_epochs = config['num_epochs']
    lr = config['lr']

    # num_epoch_dict = config['hparams'].get('num_epochs')
    # if num_epoch_dict is not None:
    #     num_epochs_low = num_epoch_dict['low']
    #     num_epochs_high = num_epoch_dict['high']

    # lr_dict = config['hparams'].get('lr')
    # if lr_dict is not None:
    #     lr_low = lr_dict['low']
    #     lr_high = lr_dict['high']

    # lr = trial.suggest_float("learning_rate",
    #                          lr_low,
    #                          lr_high,
    #                          log=True)

    # num_epochs = trial.suggest_int("num_epochs",
    #                                num_epochs_low,
    #                                num_epochs_high)



    # define the optimizer and loss fct
    optimizer = AdamW(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2,
                                                gamma=0.1)

    loss_fct = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        ignore_index=user_cfg["model"]["ignore_index"]
    )

    # define the train and eval metric containers
    train_metric = evaluate.load("seqeval",
                                 scheme="IOBES",
                                 mode="strict",
                                 experiment_id="train",
                                 keep_in_memory=True)
    eval_metric = evaluate.load("seqeval",
                                scheme="IOBES",
                                mode="strict",
                                experiment_id='eval',
                                keep_in_memory=True)

    model.to(device)  # move to GPU if available

    print("*" * 50)
    print(f'Running Training Loop for {num_epochs} epochs!')
    print("*" * 50)

    model.train()
    train_metrics_list, eval_metrics_list = [], []
    for epoch in range(num_epochs):
        model.train()
        for train_batch in tqdm(train_dataloader,
                                desc='train_loop'):
            run_train_loop(train_batch, model, optimizer, scheduler, loss_fct,
                           device, train_metric, label_list, user_cfg)

        # run eval loop
        run_eval_loop(eval_dataloader, model, device,
                      eval_metric, label_list, user_cfg)

        # compute the metrics at the end of each epoch
        train_metrics = utils.compute_metrics(train_metric)
        eval_metrics = utils.compute_metrics(eval_metric)

        train_metrics['epoch'] = epoch
        train_metrics_list.append(train_metrics)

        eval_metrics['epoch'] = epoch
        eval_metrics_list.append(eval_metrics)

        eval_f1 = eval_metrics['overall_f1']
        print(f"eval_f1={eval_f1}")
        # trial.report(eval_f1, epoch)

        # # Check if the trial should be pruned
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

    # Report the results to Ray Tune
    tune.report(eval_f1=eval_f1)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config file specifiying user params",
    )

    args = parser.parse_args()

    print("*" * 50)
    print('Processing...')
    print("*" * 50)

    with open(args.config, "r") as fh:
        user_cfg = yaml.safe_load(fh)

    collateral_dir = user_cfg['model']['collateral_dir']

    # if data split is provided then we're running training curve exps,
    # create the associate collateral dir
    data_split = user_cfg['data']['data_split']
    if data_split:
        collateral_dir = os.path.join(collateral_dir, f"num_contracts-{data_split}")

    if not os.path.exists(collateral_dir):
        os.makedirs(collateral_dir, exist_ok=True)

    model_dir = os.path.join(collateral_dir, 'ckpt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    user_cfg['model']['model_savepath'] = \
        os.path.join(model_dir,
                        user_cfg['model']['model_savepath'])

    # create study name using available hparams that are being tuned
    num_epochs_dict = user_cfg['hparams'].get('num_epochs')
    if num_epochs_dict is not None:
        num_epochs_low = num_epochs_dict['low']
        num_epochs_high = num_epochs_dict['high']
        user_cfg['study_name'] = f"num_epochs_{num_epochs_low}-{num_epochs_high}_{user_cfg['study_name']}"

    lr_dict = user_cfg['hparams'].get('lr')
    if lr_dict is not None:
        lr_low = lr_dict['low']
        lr_high = lr_dict['high']
        user_cfg['study_name'] = f"lr_{lr_low}-{lr_high}_{user_cfg['study_name']}"

    # study = optuna.create_study(direction="maximize",
    #                             study_name=config['study_name'])

    # study.optimize(lambda trial: objective(trial, config), n_trials=10)

    # # Get the best hyperparameters
    # best_trial = study.best_trial
    # best_hparams = best_trial.params

    # # Save the best hyperparameters to a JSON file
    # outpath = os.path.join(config['model']['collateral_dir'],
    #                        "best_hparams.json")
    # with open(outpath, "w") as f:
    #     json.dump(best_hparams, f)

    # Define the search space
    config = {
        "lr": tune.loguniform(lr_low,
                              lr_high),
        "num_epochs": tune.choice([2, 4, 6])
}
    algo = OptunaSearch()

    config['user_cfg'] = user_cfg

    # Run the hyperparameter search
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(objective),
            # lambda config, user_cfg: objective(config, user_cfg),
            resources={"cpu": 4, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
        metric="eval_f1",
        mode="max",
        search_alg=algo,
        num_samples=3,
    ),
        param_space=config,
    )

    results = tuner.fit()
    # Get the best hyperparameters

    best_config = results.get_best_result().config

    # Save the best hyperparameters to a JSON file
    outpath = os.path.join(user_cfg['model']['collateral_dir'],
                           "best_hparams.json")
    with open(outpath, "w") as f:
        json.dump(best_config, f)
