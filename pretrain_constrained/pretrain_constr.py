#! /usr/bin/env python3

import logging, sys
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs

if __name__ == "__main__":
	task = sys.argv[1]
	vocsize = int(sys.argv[2])

	logging.basicConfig(level=logging.INFO)
	transformers_logger = logging.getLogger("transformers")
	transformers_logger.setLevel(logging.WARNING)

	outdir_name = task + "-constr-uncased-{:d}k".format(vocsize/1000)
	model_args = LanguageModelingArgs()
	model_args.reprocess_input_data = True
	model_args.output_dir = outdir_name
	model_args.best_model_dir = outdir_name + "/best_model"
	model_args.tensorboard_dir = outdir_name + "/runs"
	model_args.overwrite_output_dir = True
	model_args.train_batch_size	= 32
	model_args.save_steps = 5000
	model_args.max_steps = 50000
	model_args.dataset_type = "simple"
	model_args.evaluate_during_training = True
	model_args.evaluate_during_training_verbose = True
	model_args.evaluate_during_training_steps = 5000
	model_args.silent = True
	model_args.do_lower_case = True
	model_args.tokenizer_name = None
	model_args.vocab_size = vocsize

	model = LanguageModelingModel("bert", None, args=model_args,
		train_files=task + "/train_textonly.txt")
	model.train_model(task + "/train_textonly.txt",
		eval_file=task + "/dev_textonly.txt")
