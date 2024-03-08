from moment_detr.inference import start_inference
from moment_detr.train import start_training, logger

if __name__ == '__main__':
    best_ckpt_path = ('results/hl-video_tef-exp_slowfast_clip_parallel_conv_prediction_head_triplet_span'
                      '-2024_03_07_00_11_08/model_best.ckpt')
    input_args = ["--resume", best_ckpt_path,
                  "--eval_split_name", 'val',
                  "--eval_id", 'val',
                  "--eval_results_dir", "results/hl-video_tef"
                                        "-exp_slowfast_clip_parallel_conv_prediction_head_triplet_span"
                                        "-2024_03_07_00_11_08/",
                  "--eval_path", "data/highlight_test_release.jsonl"]

    import sys

    sys.argv[1:] = input_args
    logger.info("Evaluating model at {}".format(best_ckpt_path))
    logger.info("Input args {}".format(sys.argv))
    start_inference()
