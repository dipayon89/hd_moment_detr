from moment_detr.inference import start_inference
from moment_detr.train import start_training, logger

if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
