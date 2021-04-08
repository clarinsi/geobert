import sys, os, logging, json, shutil, pickle, re
import numpy as np
import pandas as pd
import torch, sklearn
import simpletransformers.classification


R = 6371


def load(filename, cut=-1, blind=False):
	x = []
	y = []
	for line in open(filename, 'r', encoding='utf-8'):
		if blind:
			x.append(line.strip())
		else:
			elements = [x.strip() for x in line.split("\t")]
			x.append(elements[2])
			y.append((float(elements[0]), float(elements[1])))
		if len(x) == cut:
			break
	if blind:
		return x
	else:
		return x, np.array(y, dtype=np.float32)


# uses dimension-specific mean, but joint stddev for scaling
class JointScaler():
	def __init__(self):
		self.means = None
		self.stddev = None

	def fit_transform(self, data):
		self.means = np.mean(data, axis=0)
		centereddata = data - self.means
		self.stddev = np.std(centereddata)
		return centereddata / self.stddev

	def transform(self, data):
		return (data - self.means) / self.stddev

	def inverse_transform(self, data):
		return (data * self.stddev) + self.means


class YNormalizer():
	def __init__(self, settings):
		print("  Output normalizer settings:", settings)
		if settings == "indscale":
			self.scale = True
			self.scaler = sklearn.preprocessing.StandardScaler()
		elif settings == "jointscale":
			self.scale = True
			self.scaler = JointScaler()
		else:
			self.scale = False
			self.scaler = None

	def fit_transform(self, data):
		if self.scale:
			data = self.scaler.fit_transform(data)
		return data

	def transform(self, data):
		if self.scale:
			data = self.scaler.transform(data)
		return data

	def inverse_transform(self, data):
		if self.scale:
			data = self.scaler.inverse_transform(data)
		return data

# define globally to use it in custom loss function
normalizer = None

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
	global normalizer
	a_tr = normalizer.inverse_transform(a)
	b_tr = normalizer.inverse_transform(b)
	d = evaluate(a_tr, b_tr)
	return np.median(d)

def mean_dist(a, b):
	global normalizer
	a_tr = normalizer.inverse_transform(a)
	b_tr = normalizer.inverse_transform(b)
	d = evaluate(a_tr, b_tr)
	return np.mean(d)

###################################

def train(modelname, args):
	global normalizer
	logging.basicConfig(level=logging.INFO)
	transformers_logger = logging.getLogger("transformers")
	transformers_logger.setLevel(logging.INFO)

	if "regression" not in args:
		args["regression"] = True
	if "_lossfn" not in args:
		args["_lossfn"] = None
	if "do_lower_case" not in args:
		args["do_lower_case"] = "uncased" in args["_model_name"]
		print("Setting do_lower_case to", args["do_lower_case"])
	if "output_dir" not in args:
		args["output_dir"] = modelname
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
	args["early_stopping_metric"] = "median_dist"
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
	normalizer = YNormalizer(args["_ynorm"])
	if "_cut" in args:
		train_x, train_y = load(args["_train_data"], cut=args["_cut"])
	else:
		train_x, train_y = load(args["_train_data"])
	train_y_sc = normalizer.fit_transform(train_y)
	nbExamples = len(train_x)
	nbLabels = train_y_sc.shape[1]
	print("  {} training examples with {} output features".format(nbExamples, nbLabels))
	train_df = pd.DataFrame(zip(train_x, train_y_sc))
	train_df.columns = ["text", "labels"]
	print("  Training dataframe shape:", train_df.shape)

	dev_x, dev_y = load(args["_dev_data"])
	dev_y_sc = normalizer.transform(dev_y)
	dev_df = pd.DataFrame(zip(dev_x, dev_y_sc))
	dev_df.columns = ["text", "labels"]
	print("  Dev dataframe shape:", dev_df.shape)

	print("Train model")
	model = simpletransformers.classification.MultiLabelClassificationModel(
		args["_model_type"], args["_model_name"], loss_fct=args["_lossfn"],
		num_labels=nbLabels, args=args, use_cuda=cuda_available
	)
	model.train_model(train_df, eval_df=dev_df,
		median_dist=median_dist, mean_dist=mean_dist
	)

	# if intermediate models are not saved, these directories do not contain anything useful
	for checkpointdir in os.listdir(args["output_dir"]):
		if checkpointdir.startswith("checkpoint-"):
			print("Remove", args["output_dir"] + "/" + checkpointdir)
			shutil.rmtree(args["output_dir"] + "/" + checkpointdir)
	pickle.dump(normalizer, open(args["output_dir"] + "/normalizer.pkl", "wb"))
	model = None
	normalizer = None
	torch.cuda.empty_cache()
	print("Training finished")


def predict(modelname, args):
	global normalizer
	if "output_dir" not in args:
		args["output_dir"] = modelname
	if "best_model_dir" not in args:
		args["best_model_dir"] = args["output_dir"] + "/best_model"

	cuda_available = torch.cuda.is_available()
	print("Using CUDA:", cuda_available)

	print("Load model")
	normalizer = pickle.load(open(args["output_dir"] + "/normalizer.pkl", "rb"))
	model = simpletransformers.classification.MultiLabelClassificationModel(
		args["_model_type"], args["best_model_dir"], use_cuda=cuda_available
	)

	print("Load test data")
	if "_test_data" in args:
		test_x = load(args["_test_data"], blind=True)
	else:
		test_x, test_y = load(args["_dev_data"])
	test_df = pd.DataFrame(test_x)
	test_df.columns = ["text"]
	print("  Test dataframe shape:", test_df.shape)

	print("Predict test output")
	predictions, _ = model.predict(test_df["text"])
	predictions_trans = normalizer.inverse_transform(predictions)
	if "_test_data" in args:
		np.savetxt(args["output_dir"] + "/pred_test.txt", predictions_trans,
			delimiter='\t', fmt="%.4f")
	else:
		np.savetxt(args["output_dir"] + "/pred_dev.txt", predictions_trans,
			delimiter='\t', fmt="%.4f")
		distances = evaluate(predictions_trans, test_y)
		median = np.median(distances)
		mean = np.mean(distances)
		print("  Median distance: {:.2f} km".format(median))
		print("  Mean distance:   {:.2f} km".format(mean))
		print("", flush=True)
	normalizer = None
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
			if "_train_data" in expargs:
				train(expname, expargs)
			else:
				print("** No training data given, skip training")
			if "_save_predictions" in expargs and expargs["_save_predictions"]:
				predict(expname, expargs)
		else:
			print("** Skipping experiment", expname)
