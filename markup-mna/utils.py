import os
import glob
import pandas as pd
import torch
import random

import input_pipeline


def ablation(config, inputs):
    if config["ablation"]["is_shuffle_xpath_exp"]:
        batch_size, shuffling_dim = inputs["input_ids"].size()

        # when shuffling the xpath tokens, we shuffle along axis 1 (n=512)
        # so that the order of the DOM tree is nullified
        for ii in range(batch_size):
            inputs["xpath_tags_seq"][ii, :, :] = inputs["xpath_tags_seq"][
                ii, torch.randperm(shuffling_dim), :
            ]

            inputs["xpath_subs_seq"][ii, :, :] = inputs["xpath_subs_seq"][
                ii, torch.randperm(shuffling_dim), :
            ]

    else:
        # setting the xpath embedding to correponding pad tokens
        xpath_tag_pad_token = config["ablation"]["xpath_tag_pad_token"]
        xpath_subs_pad_token = config["ablation"]["xpath_subs_pad_token"]

        # set the xpath embeddings to the pad token
        inputs["xpath_tags_seq"].fill_(xpath_tag_pad_token)
        inputs["xpath_subs_seq"].fill_(xpath_subs_pad_token)

    return inputs


def get_label_list(config):
    # we have beis labels for title, section title/number,
    # subsection title/number, subsubsection title/number and page number + 1
    # for outside
    label_list = config["data"]["label_list"]

    # create the mapping between labels and label ids and the reverse mapping
    id2label = {idx: label for idx, label in enumerate(label_list)}
    label2id = {label: id for id, label in id2label.items()}

    # convert the label list to the format required for the huggingface metrics
    # library
    label_list = [x.replace("_", "-").upper() for x in label_list]

    return label_list, id2label, label2id


def get_dataset(contract_dir, id2label, label2id,
                data_split, num_contracts=None):
    # get the list of contracts in the provided dir
    contract_dir = os.path.join(contract_dir, "*.csv")
    contracts_list = glob.glob(contract_dir)

    print("*" * 50)
    print("length contract list", len(contracts_list))
    print("*" * 50)

    # if data split is provided then shuffle and select the first data_split
    # entries
    if num_contracts:
        random.seed(42)
        random.shuffle(contracts_list)

        contracts_list = contracts_list[:num_contracts]

    print(f"Num contracts in {data_split}: {len(contracts_list)}; num_contracts arg: {num_contracts} ")

    data = []
    for tagged_path in contracts_list:
        try:
            tagged_output = input_pipeline.create_raw_dataset(
                tagged_path, id2label=id2label, label2id=label2id
            )

            data.append(tagged_output)
        except Exception as e:
            print(f"tagged_path={tagged_path}")
            continue

    return data


def get_class_dist(contract_dir, id2label, label2id,
                   mode='inverse'):
    # get the list of contracts in the provided dir
    contracts_list = glob.glob(os.path.join(contract_dir, "*.csv"))

    list_of_tagged_df_labels = [
        pd.read_csv(x).loc[:, ["tagged_sequence"]] for x in contracts_list
    ]
    consolidated_tagged_df_labels = pd.concat(list_of_tagged_df_labels, axis=0)

    class_value_counts = consolidated_tagged_df_labels["tagged_sequence"].value_counts()
    total_examples = sum(class_value_counts.to_dict().values())

    class_weights = torch.zeros(len(label2id))

    # if in inverse mode then compute the ratio of
    # total_examples / num_ex_per_class else compute the inverse of this ratio
    if mode == 'inverse':
        for label_name, label_id in label2id.items():
            # if no examples for that class are present
            # then the weight of that class is 1
            class_count = class_value_counts.get(label_name, total_examples)
            class_weights[label_id] = total_examples / class_count
    elif mode == 'normal':
        for label_name, label_id in label2id.items():
            # if no examples for that class are present
            # then the weight of that class is 0
            class_count = class_value_counts.get(label_name, 0)
            class_weights[label_id] = class_count / total_examples

    return class_value_counts, class_weights


def convert_preds_to_labels(predictions, references,
                            label_list, device="cpu",
                            ignore_index=-100):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != ignore_index]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != ignore_index]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels


def compute_metrics(metric, return_entity_level_metrics=True):
    results = metric.compute()
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
