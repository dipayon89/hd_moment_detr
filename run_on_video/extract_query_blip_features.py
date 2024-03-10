import numpy as np
import torch
from os.path import join

import json

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
# text_input = txt_processors["eval"](caption)
# sample = {"image": image, "text_input": [text_input]}

def encode_text_query(batch):
    batch_output = []
    with torch.no_grad():
        for text in batch:
            text_input = txt_processors["eval"](text)
            sample = { "text_input": [text_input]}
            features_text = model.extract_features(sample, mode="text")
            batch_output.append(features_text.text_embeds.squeeze())
        return batch_output

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


class QVHighlightsDataset(Dataset):
    def __init__(self, input_file):
        self.datalist = load_jsonl(input_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, i):
        # query = self.datalist[i]["query"]
        # prompt = f'[INST]Paraphrase the text in quatation mark. "{query}"[/INST]\n'
        # return prompt
        new_dict = dict.fromkeys(
            ['qid', 'query', 'duration', 'vid', 'relevant_clip_ids', 'saliency_scores', 'relevant_windows'])
        new_dict.update(self.datalist[i])
        return new_dict


def generate_batched_query(batch):
    # print(batch)
    return batch['query']


def save_query_features(batch, batch_result, q_feat_dir, training=True):
    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        if training:
            aug_id = batch["aug_id"][i]
        else:
            aug_id = 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")
        np.savez_compressed(q_feat_path, last_hidden_state=result[:-1].cpu())


def collate_fn(batch):
    """Collates a batch of dictionaries into a single dictionary.

    Args:
      batch: A list of dictionaries.

    Returns:
      A single dictionary.
    """

    collated_dict = {}
    for key in batch[0]:
        collated_dict[key] = [data[key] for data in batch]
    return collated_dict


def extract_train_query_features():
    input_file = "../data/highlight_train_release_paraphrased_openai.jsonl"
    q_feat_dir = "../../QVHighlights/features/blip_aug_text_features_openai"

    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        # print(batch_prompt)
        batch_result = encode_text_query(batch_query)
        # print(batch_result)
        save_query_features(batch, batch_result, q_feat_dir)


def extract_val_query_features():
    input_file = "../data/highlight_val_release.jsonl"
    q_feat_dir = "../../QVHighlights/features/blip_aug_text_features_openai"

    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        # print(batch_prompt)
        batch_result = encode_text_query(batch_query)
        # print(batch_result)
        save_query_features(batch, batch_result, q_feat_dir, False)

def extract_test_query_features():
    input_file = "../data/highlight_test_release.jsonl"
    q_feat_dir = "../../QVHighlights/features/blip_aug_text_features_openai"

    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        # print(batch_prompt)
        batch_result = encode_text_query(batch_query)
        # print(batch_result)
        save_query_features(batch, batch_result, q_feat_dir, False)

def extract_all_query_features():
    extract_train_query_features()
    extract_val_query_features()
    extract_test_query_features()

if __name__ == "__main__":
    extract_all_query_features()
    # x = feature_extractor.encode_text_query(["Chef makes pizza and cuts it up.", "Chef makes pizza and cuts"])
    # print(x)
