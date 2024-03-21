import json
import os
from os.path import join

import torch
from lavis.models import load_model_and_preprocess

from run_on_video.data_utils import VideoLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
#                                                                   model_type="pretrain", is_eval=True,
#                                                                   device=device)  # Blip2 featrures
# {
#   "qid": "x68guk71VFo_360.0_510.0_subs56",
#   "query": "yeah",
#   "vid": "x68guk71VFo_360.0_510.0",
#   "duration": 150,
#   "split": "train",
#   "relevant_windows": [[97.919, 98.39]]
# }

# we associate a model with its preprocessors to make it easier for inference.
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip_caption", model_type="large_coco", is_eval=True, device=device
# )
# uncomment to use base model
model, vis_processors, _ = load_model_and_preprocess(
    name="blip_caption", model_type="base_coco", is_eval=True, device=device
)
vis_processors.keys()

# we associate a model with its preprocessors to make it easier for inference.
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
# )

# Other available models:
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
# )
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
# )
#
# model, vis_processors, _ = load_model_and_preprocess(
#     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
# )

video_loader = VideoLoader(framerate=0.1, size=224, centercrop=True)

v_input_dir = "../QVHighlights/processed_videos/"
pretrain_jsonl_path = "data/highlight_train_release.jsonl"


@torch.no_grad()
def encode_video(input_dir: str, vid: str):
    video_path = join(input_dir, f"{vid}.mp4")
    video_frames, info = video_loader.read_raw_image_from_video_file(video_path)  # (T, H, W, 3)
    split = "train"
    duration = info["duration"]
    n_frames = len(video_frames)
    frame_duration = duration / n_frames
    start_duration = 0
    train_data = []
    for i in range(n_frames):
        qid = f"{vid}_{start_duration}_{start_duration + frame_duration}"
        image = vis_processors["eval"](video_frames[i]).unsqueeze(0).to(device)
        query = model.generate({"image": image})
        relevant_windows = [[start_duration, start_duration + frame_duration]]
        start_duration += frame_duration
        dict_data = {
            "qid": qid,
            "query": query,
            "vid": vid,
            "duration": duration,
            "split": split,
            "relevant_windows": relevant_windows
        }
        train_data.append(dict_data)
    return train_data


def read_all_files_from_directory(directory_path):
    return [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]


def generate_pretrain_data(input_dir):
    train_data = []
    video_files = read_all_files_from_directory(input_dir)
    with torch.no_grad():
        for video in video_files:
            train_data.extend(encode_video(input_dir,
                                           os.path.splitext(video)[0]))
    return train_data


def extract_and_load_pretrain_data():
    train_data = generate_pretrain_data(v_input_dir)
    with open(pretrain_jsonl_path, "w") as f:
        for data in train_data:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    extract_and_load_pretrain_data()
