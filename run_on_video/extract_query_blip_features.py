import io
import math
import os

import numpy as np
import torch
from os.path import join

import json

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from lavis.models import load_model_and_preprocess

from run_on_video.data_utils import VideoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base",
                                                                  is_eval=True, device=device)
# text_input = txt_processors["eval"](caption)
# sample = {"image": image, "text_input": [text_input]}
video_loader = VideoLoader(framerate=0.5, size=224, centercrop=True)

v_input_dir = "../QVHighlights/processed_videos/"
v_feat_dir = "../QVHighlights/features/blip_video_features/"
q_feat_dir = "../QVHighlights/features/blip_aug_text_features_openai"

def encode_text_query(batch):
    batch_output = []
    with torch.no_grad():
        for text in batch:
            text_input = txt_processors["eval"](text)
            sample = {"text_input": [text_input]}
            features_text = model.extract_features(sample, mode="text")
            batch_output.append(features_text)
        return batch_output


def encode_video_query(input_dir, batch):
    batch_output = []
    with torch.no_grad():
        for vid in batch:
            video_path = join(input_dir, f"{vid}.mp4")
            # print("video_path", video_path)
            video_feature = encode_video(video_path)
            batch_output.append(video_feature)
        return batch_output


@torch.no_grad()
def encode_video(video_path: str):
    video_frames = video_loader.read_raw_image_from_video_file(video_path)  # (T, H, W, 3)
    n_frames = len(video_frames)
    video_features = []
    for i in range(n_frames):
        image = vis_processors["eval"](video_frames[i]).unsqueeze(0).to(device)
        sample = {"image": image}
        features_image = model.extract_features(sample, mode="image")
        video_features.append(features_image.image_embeds[:, 0, :])
    video_features = torch.cat(video_features, dim=0)
    return video_features  # (T=#frames, d) torch tensor


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


def generate_batched_vid(v_feat_dir, batch):
    return [vid for vid in batch['vid'] if not is_file_present(v_feat_dir, vid)]


def save_query_features(batch, batch_result, q_feat_dir, training=True):
    for i, result in enumerate(batch_result):
        qid = batch["qid"][i]
        if training:
            aug_id = batch["aug_id"][i]
        else:
            aug_id = 0
        aug = f"_{aug_id}" if aug_id > 0 else ""
        q_feat_path = join(q_feat_dir, f"qid{qid}{aug}.npz")
        pooler_output = result.text_embeds[:, 0, :].squeeze()
        np.savez_compressed(q_feat_path, last_hidden_state=result.text_embeds.squeeze().cpu(),
                            pooler_output=pooler_output.cpu())


def save_video_features(batch, batch_result, v_feat_dir):
    for i, result in enumerate(batch_result):
        vid = batch["vid"][i]
        v_feat_path = join(v_feat_dir, f"{vid}.npz")
        np.savez_compressed(v_feat_path, features=result.cpu())



# write a code to check if a file is present in the directory
# if not present, then only extract the features
def is_file_present(v_feat_dir, vid):
    file_path = join(v_feat_dir, f"{vid}.npz")
    return os.path.exists(file_path)


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


def extract_video_features(input_file):
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_vid = generate_batched_vid(v_feat_dir, batch)
        if len(batch_vid) == 0:
            print("All files present:", batch_vid)
            continue
        print("Processing:", batch_vid)
        batch_result = encode_video_query(v_input_dir, batch_vid)
        # print(batch_result)
        save_video_features(batch, batch_result, v_feat_dir)


def extract_train_video_features():
    input_file = "data/highlight_train_release.jsonl"
    extract_video_features(input_file)



def extract_val_video_features():
    input_file = "data/highlight_val_release.jsonl"
    extract_video_features(input_file)


def extract_test_video_features():
    input_file = "data/highlight_test_release.jsonl"
    extract_video_features(input_file)


def extract_query_features(input_file):
    dataset = QVHighlightsDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=collate_fn)

    for batch in tqdm(dataloader):
        batch_query = generate_batched_query(batch)
        # print(batch_prompt)
        batch_result = encode_text_query(batch_query)
        # print(batch_result)
        save_query_features(batch, batch_result, q_feat_dir)


def extract_train_query_features():
    input_file = "data/highlight_train_release_paraphrased_openai.jsonl"
    extract_query_features(input_file)



def extract_val_query_features():
    input_file = "data/highlight_val_release.jsonl"
    extract_query_features(input_file)


def extract_test_query_features():
    input_file = "data/highlight_test_release.jsonl"
    extract_query_features(input_file)


def extract_all_query_features():
    # extract_train_query_features()
    # extract_val_query_features()
    # extract_test_query_features()
    extract_train_video_features()
    extract_val_video_features()
    extract_test_video_features()


if __name__ == "__main__":
    extract_all_query_features()
    # x = feature_extractor.encode_text_query(["Chef makes pizza and cuts it up.", "Chef makes pizza and cuts"])
    # print(x)
