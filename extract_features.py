from moment_detr.inference import start_inference
from moment_detr.train import start_training, logger
# from run_on_video.extract_query_clip_features import extract_all_query_features
from run_on_video.extract_query_blip_features_tvsum import extract_all_query_features

if __name__ == '__main__':
    extract_all_query_features()
