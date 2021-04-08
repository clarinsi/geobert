import sys, os, logging, json, shutil, pickle, re
import numpy as np
import pandas as pd
import torch, sklearn
import simpletransformers.classification


R = 6371


def load(textfilename, labelfilename, cut=-1, blind=False):
	x = []
	y = []
	textfile = open(textfilename, 'r', encoding='utf-8')
	if blind:
		for line in textfile:
			x.append(line.strip())
		return x
	elif labelfilename is None:	# use original coords from textfilename
		for line in textfile:
			elements = line.split("\t")
			text = elements[-1].strip()
			coords = (float(elements[0]), float(elements[1]))
			x.append(text)
			y.append(coords)
			if len(x) == cut:
				break
		return x, np.array(y)
	else:
		labelfile = open(labelfilename, 'r', encoding='utf-8')
		for textline, labelline in zip(textfile, labelfile):
			text = textline.split("\t")[-1].strip()
			label = labelline.strip()
			x.append(text)
			y.append(label)
			if len(x) == cut:
				break
		return x, y


def labels2array(l):
	global ids2labels
	labels = [ids2labels[x] for x in l]
	return np.array([
		(float(x.split("_")[0]), float(x.split("_")[1])) for x in labels
	])


# c1 = [lat, lon], c2 = [lat, lon]
def evaluate(c1, c2, scale_km=True):
	d = np.radians(c2-c1)
	a = np.sin(d[:,0]/2) * np.sin(d[:,0]/2) + np.cos(np.radians(c1[:,0])) * np.cos(np.radians(c2[:,0])) * np.sin(d[:,1]/2) * np.sin(d[:,1]/2)
	d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
	if scale_km:
		return R * d
	else:
		return d

###################################

def median_dist(a, b):
	a_tr = labels2array(a)
	b_tr = labels2array(b)
	d = evaluate(a_tr, b_tr)
	return np.median(d)

def mean_dist(a, b):
	a_tr = labels2array(a)
	b_tr = labels2array(b)
	d = evaluate(a_tr, b_tr)
	return np.mean(d)

###################################

def train(modelname, args):
	logging.basicConfig(level=logging.INFO)
	transformers_logger = logging.getLogger("transformers")
	transformers_logger.setLevel(logging.INFO)

	if "regression" not in args:
		args["regression"] = False
	if "do_lower_case" not in args:
		args["do_lower_case"] = "uncased" in args["_model_name"]
		print("Setting do_lower_case to", args["do_lower_case"])
	if "output_dir" not in args:
		args["output_dir"] = modelname
		if "seed" in args:
			args["output_dir"] += "_{}".format(args["seed"])
	args["best_model_dir"] = args["output_dir"] + "/best_model"
	args["tensorboard_dir"] = args["output_dir"] + "/runs"
	args["overwrite_output_dir"] = True
	if "silent" not in args:
		args["silent"] = True
		# hides progress bars which are inconvenient when redirecting output to file
	args["evaluate_during_training"] = True
	args["evaluate_during_training_verbose"] = True
	if "evaluate_during_training_steps" not in args:
		args["evaluate_during_training_steps"] = 0

	# save only the best model, not any other intermediate model
	# this does not actually do early stopping as long as use_early_stopping is not set to True
	args["early_stopping_metric"] = "eval_loss" # "median_dist"
	args["early_stopping_metric_minimize"] = True
	args["save_eval_checkpoints"] = False
	args["save_model_every_epoch"] = False

	print("Settings:")
	print(json.dumps(args, indent=2))
	if args["output_dir"] not in os.listdir("."):
		os.mkdir(args["output_dir"])
	if "seed" in args:
		print("Setting random seed:", args["seed"])
		torch.manual_seed(int(args["seed"]))

	cuda_available = torch.cuda.is_available()
	print("Using CUDA:", cuda_available)

	print("Load data")
	if "_cut" in args:
		train_x, train_y = load(args["_train_data"], args["_train_labels"],
			cut=args["_cut"])
	else:
		train_x, train_y = load(args["_train_data"], args["_train_labels"])
	nbExamples = len(train_x)

	# create our own label conversion, the built-in one doesn't seem to work
	global labels2ids, ids2labels
	labellist = list(set(train_y))
	labels2ids = {l: i for i, l in enumerate(labellist)}
	ids2labels = {i: l for i, l in enumerate(labellist)}
	print("labels2ids", labels2ids)
	print("ids2labels", ids2labels)
	train_y_ids = [labels2ids[y] for y in train_y]
	print("  {} training examples with {} output features".format(nbExamples, len(labellist)))
	train_df = pd.DataFrame(zip(train_x, train_y_ids))
	train_df.columns = ["text", "labels"]
	print("  Training dataframe shape:", train_df.shape)
	print(train_df.head())

	dev_x, dev_y = load(args["_dev_data"], args["_dev_labels"])
	dev_y_ids = [labels2ids[y] for y in dev_y]
	dev_df = pd.DataFrame(zip(dev_x, dev_y_ids))
	dev_df.columns = ["text", "labels"]
	print("  Dev dataframe shape:", dev_df.shape)

	print("Train model")
	model = simpletransformers.classification.ClassificationModel(
		args["_model_type"], args["_model_name"], args=args,
		num_labels=len(labellist), use_cuda=cuda_available
	)
	model.train_model(train_df, eval_df=dev_df,
		median_dist=median_dist, mean_dist=mean_dist
	)

	# if intermediate models are not saved, these directories do not contain anything useful
	for checkpointdir in os.listdir(args["output_dir"]):
		if checkpointdir.startswith("checkpoint-"):
			print("Remove", args["output_dir"] + "/" + checkpointdir)
			shutil.rmtree(args["output_dir"] + "/" + checkpointdir)
	pickle.dump(ids2labels, open(args["output_dir"] + "/ids2labels.pkl", "wb"))
	model = None
	torch.cuda.empty_cache()
	print("Training finished")


def predict(modelname, args):
	if "output_dir" not in args:
		args["output_dir"] = modelname
	if "best_model_dir" not in args:
		args["best_model_dir"] = args["output_dir"] + "/best_model"

	cuda_available = torch.cuda.is_available()
	print("Using CUDA:", cuda_available)

	print("Load model")
	global ids2labels
	ids2labels = pickle.load(open(args["output_dir"] + "/ids2labels.pkl", "rb"))
	model = simpletransformers.classification.ClassificationModel(
		args["_model_type"], args["best_model_dir"],
		num_labels=len(ids2labels), use_cuda=cuda_available
	)

	print("Load test data")
	if "_test_data" in args:
		test_x = load(args["_test_data"], None, blind=True)
	else:
		test_x, test_y_real = load(args["_dev_data"], None)
	test_df = pd.DataFrame(test_x)
	test_df.columns = ["text"]
	print("  Test dataframe shape:", test_df.shape)

	print("Predict test output")
	predictions, _ = model.predict(test_df["text"])
	pred_array = labels2array(predictions)
	if "_test_data" in args:
		np.savetxt(args["output_dir"] + "/pred_test.txt", pred_array,
			delimiter='\t', fmt="%.4f")
	else:
		np.savetxt(args["output_dir"] + "/pred_dev.txt", pred_array,
			delimiter='\t', fmt="%.4f")
		distances = evaluate(pred_array, test_y_real)
		median = np.median(distances)
		mean = np.mean(distances)
		print("  Median distance: {:.2f} km".format(median))
		print("  Mean distance:   {:.2f} km".format(mean))
		print("", flush=True)
	model = None
	torch.cuda.empty_cache()

if __name__ == "__main__":
	experiments = json.load(open(sys.argv[1], "r"))
	if len(sys.argv) == 3:
		filter = sys.argv[2]
	else:
		filter = r'.*'
	for expname, expargs in experiments.items():
		if re.search(filter, expname):
			print("**", expname, "**")
			if "_predict_only" not in expargs or not expargs["_predict_only"]:
				train(expname, expargs)
			if "_save_predictions" in expargs and expargs["_save_predictions"]:
				predict(expname, expargs)
		else:
			print("** Skipping experiment", expname)
