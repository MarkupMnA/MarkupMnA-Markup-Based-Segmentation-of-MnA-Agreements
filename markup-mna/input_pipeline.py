import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import pandas as pd
from torch.utils.data import Dataset


class MarkupLMDataset(Dataset):
    """Dataset for token classification with MarkupLM."""

    def __init__(self, data, processor=None, max_length=512):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.encodings = []
        self.get_encoding_windows()

    def get_encoding_windows(self):
        """Splits the tokenized input into windows of 512 tokens"""

        for item in self.data:
            nodes, xpaths, node_labels = (
                item["nodes"],
                item["xpaths"],
                item["node_labels"],
            )

            # provide to processor
            encoding = self.processor(
                nodes=nodes,
                xpaths=xpaths,
                node_labels=node_labels,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                truncation=False,
            )

            # remove batch dimension
            encoding = {k: v.squeeze() for k, v in encoding.items()}

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


def create_raw_dataset(tagged_csv_path, id2label, label2id):
    """Preprocesses the tagged csvs in the format required by MarkupLM"""
    col_list = ["nodes", "xpaths", "node_labels"]

    tagged_df = pd.read_csv(tagged_csv_path)

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

    tagged_output = tagged_df.loc[:, col_list].to_dict(orient="list")

    # convert each key to a list of lists just like the MarkupLM
    # pipeline requires
    for k, v in tagged_output.items():
        tagged_output[k] = [v]

    return tagged_output
