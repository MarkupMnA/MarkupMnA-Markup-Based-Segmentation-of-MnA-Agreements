import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import os 
import glob
import json
from transformers import MarkupLMProcessor
from transformers import MarkupLMForTokenClassification

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import utils
# import input_pipeline as ip

import pandas as pd

from tqdm.auto import tqdm
import yaml
from IPython.display import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

def create_raw_dataset(tagged_csv_path, id2label, label2id, is_train=True):
    """Preprocesses the tagged csvs in the format required by MarkupLM"""

    tagged_df = pd.read_csv(tagged_csv_path)

    # in train mode we expect text and xpaths that are highlighted 
    # by an annotator
    if is_train:
        col_list = ["nodes", "xpaths", "node_labels"]
        
        tagged_df["highlighted_xpaths"] = tagged_df["highlighted_xpaths"].fillna(
            tagged_df["xpaths"]
        )
        tagged_df["highlighted_segmented_text"] = tagged_df[
            "highlighted_segmented_text"
        ].fillna(tagged_df["text"])

        # drop non-ASCII chars
        tagged_df["highlighted_segmented_text"] = (
            tagged_df["highlighted_segmented_text"]
            .str.encode("ascii", errors="ignore")
            .str.decode("ascii")
        )

        # rename columns to match MarkupLM convention
        tagged_df = tagged_df.rename(
            columns={
                "highlighted_xpaths": "xpaths",
                "highlighted_segmented_text": "nodes",
                "tagged_sequence": "node_labels",
            },
        )

        # convert node labels to integer values
        tagged_df["node_labels"] = tagged_df["node_labels"].apply(
            lambda label: label2id[label]
        )
    
    else:
        col_list = ["nodes", "xpaths"]
        
        # rename columns to match MarkupLM convention
        tagged_df = tagged_df.rename(
            columns={
                "xpaths": "xpaths",
                "text": "nodes",
            },
        )     
    
    tagged_output = tagged_df.loc[:, col_list].to_dict(orient="list")

    # convert each key to a list of lists just like the MarkupLM
    # pipeline requires
    for k, v in tagged_output.items():
        tagged_output[k] = [v]

    return tagged_output

def create_raw_dataset_any_html(html_all_nodes_path, id2label=None, label2id=None, is_train=False):
    """Preprocesses the tagged csvs in the format required by MarkupLM"""
    import ast
    
    with open(html_all_nodes_path, 'r') as f:
        test_json = json.load(f)
    
    texts = ast.literal_eval(test_json['segmentedTexts'])
    xpaths = ast.literal_eval(test_json['xpaths'])
    tagged_df = pd.DataFrame.from_dict({'nodes':texts, 'xpaths':xpaths})
    tagged_df = tagged_df.where(tagged_df['xpaths'] != '').dropna()
    col_list = ["nodes", "xpaths"]
    
    tagged_output = tagged_df.loc[:, col_list].to_dict(orient="list")

    # convert each key to a list of lists just like the MarkupLM
    # pipeline requires
    for k, v in tagged_output.items():
        tagged_output[k] = [v]

    return tagged_df, tagged_output

class MarkupLMDataset(Dataset):
    """Dataset for token classification with MarkupLM."""

    def __init__(self, data, processor=None, max_length=512, is_train=True):
        self.data = data
        self.is_train = is_train
        self.processor = processor
        self.max_length = max_length
        self.encodings = []
        self.get_encoding_windows()
        

    def get_encoding_windows(self):
        """Splits the tokenized input into windows of 512 tokens"""
          
        for item in self.data:            
            if self.is_train:
                nodes, xpaths, node_labels = (
                    item["nodes"],
                    item["xpaths"],
                    item['node_labels']
                )
            else:
                
                nodes, xpaths, node_labels = (
                    item["nodes"],
                    item["xpaths"],
                    None
                )                

                
            # provide encoding to processor
            encoding = self.processor(
                nodes=nodes,
                xpaths=xpaths,
                node_labels=node_labels,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                truncation=False,
                return_offsets_mapping=True
            )

            # remove batch dimension
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            
            # chunk up the encoding sequences to that it is less than the 
            # max input length of 512 tokens
            if not self.is_train:
                num_tokens = encoding['input_ids'].size(-1)

                for idx in range(0, num_tokens, self.max_length):
                    batch_encoding = {}
                    for k, v in encoding.items():
                        batch_encoding[k] = v[idx: idx + self.max_length]

                    self.encodings.append(batch_encoding)                    
                    continue
            
            else:
                if len(encoding["input_ids"]) <= self.max_length:                    
                    self.encodings.append(encoding)
                    continue

                else:
                    batch_encoding = {}

                    start_idx, end_idx = 0, self.max_length

                    while end_idx < len(encoding["input_ids"]):
                        # decrement the end_idx by 1 until the label is not -100
                        while encoding["labels"][end_idx] == -100:
                            end_idx = end_idx - 1

                            # if the end idx is equal to the start idx meaning
                            # we don't encounter a non -100 token,
                            # we set window size as the max_length
                            if end_idx == start_idx:
                                end_idx = start_idx + self.max_length
                                break

                        for k, v in encoding.items():
                            batch_encoding[k] = v[start_idx:end_idx]

                        self.encodings.append(batch_encoding)
                        batch_encoding = {}

                        # update the pointers
                        start_idx = end_idx
                        end_idx = end_idx + self.max_length

                    # collect the remaining tokens
                    for k, v in encoding.items():
                        batch_encoding[k] = v[start_idx:]

                    if batch_encoding:
                        self.encodings.append(batch_encoding)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        # first, get nodes, xpaths and node labels
        item = self.encodings[idx]

        # pad the encodings to max_length of 512 tokens
        padded_item = self.processor.tokenizer.pad(
            item, max_length=self.max_length, padding="max_length", return_tensors="pt"
        )

        return padded_item
    

def run_inference_loop(dataloader, model, device, config, processor):
    '''Runs eval loop for entire dataset

    Args:
        dataloader: torch.utils.data.DataLoader: iterator over Dataset object
        model: transformers.PreTrainedModel. fine-tuned MarkupLM model
        device: torch.device. Specifies whether GPU is available for computation
        label_list: list. List of labels used to train the MarkupLM model
        config: dict. Contains user-provided params and args

    Returns:
        None
    '''
    model.eval()
    
    results = {"nodes": [], "preds": []}
    for batch in tqdm(dataloader, desc='inference_loop'):
        # get the inputs;
        inputs = {k: v.to(device) for k, v in batch.items()}

        # if ablation mode is set to true then
        # either mask the xpaths or shuffle them
        if config["ablation"]["run_ablation"]:
            inputs = utils.ablation(config, inputs)

        # get the offset mapping. It contains the spans of the 
        # words that were split during tokenization. 
        # Info present at a token level
        offset_mapping = inputs.pop('offset_mapping').squeeze().tolist()

        # forward + backward + optimize
        outputs = model(**inputs)

        predictions = outputs.logits.argmax(dim=-1)
        pred_labels = [model.config.id2label[id] for id in predictions.squeeze().tolist()]
        
        input_ids = inputs['input_ids'].detach().numpy().flatten().tolist()
        input_word_pieces = [processor.decode([id]) for id in input_ids]
        
        
        # input_ids = [x for x in input_ids if x not in special_tokens]
        # print(input_ids)
        results['nodes'].append(input_word_pieces)
        results['preds'].append(pred_labels)
                                
    return results


def main(config, test_data, model_ckpt_path=None, is_train=False):
    '''Main execution of script'''
    # get the  list of labels along with the label to id mapping and
    # reverse mapping
    label_list, id2label, label2id = utils.get_label_list(config)
    
    # define the processor and model
    if config["model"]["use_large_model"]:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-large",
            only_label_first_subword=config['model']['label_only_first_subword']
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-large", id2label=id2label, label2id=label2id
        )

    else:
        processor = MarkupLMProcessor.from_pretrained(
            "microsoft/markuplm-base",
            only_label_first_subword=config['model']['label_only_first_subword'],
        )
        model = MarkupLMForTokenClassification.from_pretrained(
            "microsoft/markuplm-base", id2label=id2label, label2id=label2id
        )
        
    if model_ckpt_path is not None:
        model_ckpt = torch.load(model_ckpt_path, 
                                map_location='cpu')
        
        model.load_state_dict(model_ckpt)
        

    processor.parse_html = False
    
    # convert the input dataset
    # to torch datasets. Create the dataloaders as well
    test_dataset = MarkupLMDataset(
        data=test_data,
        processor=processor,
        max_length=config["model"]["max_length"],
        is_train=is_train
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    model.to(device)  # move to GPU if available

    print("*" * 50)
    print(f'Running Inference Loop!')
    print("*" * 50)

    # run inference loop
    results = run_inference_loop(test_dataloader, model, device, 
                                     config, processor)


    return results

config_path = '/Users/rohith/Documents/Independent Study - DSGA1006/MarkupMnA-Markup-Based-Segmentation-of-MnA-Agreements/markup-mna/configs/config.yaml'

test_contract_dir = "/Users/rohith/Documents/Independent Study - DSGA1006/MarkupMnA-Markup-Based-Segmentation-of-MnA-Agreements/contracts/test"

# if loading from ckpt then change the line below
model_ckpt_path = "/Users/rohith/Documents/Independent Study - DSGA1006/MarkupMnA-Markup-Based-Segmentation-of-MnA-Agreements/pretrained_models/markuplm_base_model_ablation_shuffle_num_contract_100pct_f1-0.871.pt"

max_length = 512
test_batch_size = 1

# read the config file 
with open(config_path, 'r') as yml:
    config = yaml.safe_load(yml)

label_list, id2label, label2id = utils.get_label_list(config)
num_labels = len(label2id)


test_contracts = glob.glob(os.path.join(test_contract_dir, "*.csv"))

test_contracts = [test_contracts[0]]

print(f"Found {len(test_contracts)} in test dir")

test_contracts[0]

ddf = pd.read_csv(test_contracts[0])

ddf.head()

test_data = [] 
for tagged_path in test_contracts:
    tagged_output = create_raw_dataset(tagged_path, 
                                       id2label=id2label, 
                                       label2id=label2id,
                                       is_train=False)

    test_data.append(tagged_output)
    

# Actual file url: https://www.sec.gov/Archives/edgar/data/1514056/000149315223024375/formsc13d.htm
html_file_path = '/Users/rohith/Documents/Independent Study - DSGA1006/MarkupMnA-Markup-Based-Segmentation-of-MnA-Agreements/test_html.htm'
html_test_nodes_path = '/Users/rohith/Documents/Independent Study - DSGA1006/MarkupMnA-Markup-Based-Segmentation-of-MnA-Agreements/test2_html_all_nodes.json'  

# Convert file to csv of xpaths and texts, no tags
dataset_df, test_dataset_output = create_raw_dataset_any_html(html_test_nodes_path)
# Run inference on the file and return df of preds
results_test_html = main(config, [test_dataset_output], model_ckpt_path=model_ckpt_path,is_train=False)

df = pd.DataFrame.from_dict(results_test_html)

print(df.head())

df = df.explode(['nodes','preds'])
display(df.iloc[-470:-410])