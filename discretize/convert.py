#! /usr/bin/env python3

import numpy as np
import os
import sklearn.cluster


def evaluate(c1, c2):
	d = np.radians(c2-c1)
	a = np.sin(d[:,0]/2) * np.sin(d[:,0]/2) + \
		np.cos(np.radians(c1[:,0])) * np.cos(np.radians(c2[:,0])) * \
		np.sin(d[:,1]/2) * np.sin(d[:,1]/2)
	d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
	return 6371 * d


def replaceUnknown(dev, train_set):
	np_train = np.array(list(train_set))
	new_dev = []
	repl = 0
	for d in dev:
		if tuple(d) in train_set:
			new_dev.append(d)
		else:
			np_d = np.array(d).reshape(1, -1)
			distances = evaluate(np_d, np_train)
			nearest = np_train[np.argmin(distances)]
			new_dev.append(nearest)
			repl += 1
	return np.array(new_dev), repl


class KMeansConverter:
	def __init__(self, k):
		self.k = k
		self.km = sklearn.cluster.KMeans(n_clusters=self.k, random_state=0)

	def train(self, data):
		self.kmres = self.km.fit(data)

	def apply(self, data):
		labels = self.km.predict(data)
		newlabels = []
		for label in labels:
			c = self.kmres.cluster_centers_[label]
			newlabels.append(c)
		return np.array(newlabels)

	def labelStrings(self, data):
		return ["{:.2f}_{:.2f}".format(row[0], row[1]) for row in data]

	def idString(self):
		return "kmeans", "k={}".format(self.k), "kmeans_{}".format(self.k)


class MiikkaConverter:
	def __init__(self, cellsizeX, cellsizeY):
		self.cellsizeX = cellsizeX
		self.cellsizeY = cellsizeY

	def train(self, data):
		pass

	def apply(self, data):
		newlabels = []
		for d in data:
			c = (round(d[0]/self.cellsizeX, 0)*self.cellsizeX, round(d[1]/self.cellsizeY, 0)*self.cellsizeY)
			newlabels.append(c)
		return np.array(newlabels)

	def labelStrings(self, data):
		if max(self.cellsizeX, self.cellsizeY) >= 1:
			return ["{:.0f}_{:.0f}".format(row[0], row[1]) for row in data]
		else:
			return ["{:.1f}_{:.1f}".format(row[0], row[1]) for row in data]

	def idString(self):
		return "miikka", "cellsize={}x{}".format(self.cellsizeX, self.cellsizeY), "miikka_{}x{}".format(self.cellsizeX, self.cellsizeY)


class TommiConverter:
	def __init__(self, numcellsX, numcellsY):
		self.numcellsX = numcellsX
		self.numcellsY = numcellsY

	def train(self, data):
		self.minX = data[:,0].min()
		self.minY = data[:,1].min()
		self.maxX = data[:,0].max()
		self.maxY = data[:,1].max()
		self.stepX = (self.maxX-self.minX)/self.numcellsX
		self.stepY = (self.maxY-self.minY)/self.numcellsY

	def apply(self, data):
		newlabels = []
		for d in data:
			for i in range(self.numcellsX):
				if d[0] < self.minX + ((i+0.5) * self.stepX):
					newX = self.minX + i*self.stepX
					break
			for i in range(self.numcellsY):
				if d[1] < self.minY + ((i+0.5) * self.stepY):
					newY = self.minY + i*self.stepY
					break
			c = (newX, newY)
			newlabels.append(c)
		return np.array(newlabels)

	def labelStrings(self, data):
		return ["{:.2f}_{:.2f}".format(row[0], row[1]) for row in data]

	def idString(self):
		return "tommi", "numcells={}x{}".format(self.numcellsX, self.numcellsY), "tommi_{}x{}".format(self.numcellsX, self.numcellsY)


def load(filename):
	x = []
	y = []
	for line in open(filename, 'r', encoding='utf-8'):
		elements = [x.strip() for x in line.split("\t")]
		x.append(elements[2])
		y.append((round(float(elements[0]), 2), round(float(elements[1]), 2)))
	return x, np.array(y, dtype=np.float32)


def save(filename, data):
	f = open(filename, 'w')
	for row in data:
		f.write(row + "\n")
	f.close()


def convert(task, method, **kwargs):
	train_x, train_y = load("../" + task + "/train.txt")
	dev_x, dev_y = load("../" + task + "/dev.txt")

	if method == "kmeans":
		conv = KMeansConverter(kwargs["k"])
	elif method == "miikka":
		conv = MiikkaConverter(kwargs["cellsizeX"], kwargs["cellsizeY"])
	elif method == "tommi":
		conv = TommiConverter(kwargs["numcellsX"], kwargs["numcellsY"])
	else:
		return

	conv.train(train_y)
	new_train_y = conv.apply(train_y)
	new_dev_y = conv.apply(dev_y)
	train_set = set([tuple(x) for x in new_train_y])
	new_dev_y, dev_repl = replaceUnknown(new_dev_y, train_set)
	train_distances = evaluate(train_y, new_train_y)
	dev_distances = evaluate(dev_y, new_dev_y)

	if "header" in kwargs and kwargs["header"]:
		print()
		print("\t".join(["Task", "Method", "Params", "Train median", "Train mean", "Dev median", "Dev mean", "Unique labels", "Repl dev labels"]))

	print("\t".join([task.upper(), conv.idString()[0], conv.idString()[1]]), end="\t")
	print("{:.2f} km\t{:.2f} km\t{:.2f} km\t{:.2f} km".format(np.median(train_distances), np.mean(train_distances), np.median(dev_distances), np.mean(dev_distances)), end="\t")
	print("{}\t{}".format(len(train_set), dev_repl))
	save("{}/{}_train.txt".format(task, conv.idString()[2]), conv.labelStrings(new_train_y))
	save("{}/{}_dev.txt".format(task, conv.idString()[2]), conv.labelStrings(new_dev_y))


def kmeans():
	# k represents the number of target clusters
	for k in (10 ,20, 35, 50, 75, 100): convert("ch", "kmeans", k=k, header=(k==10))
	for k in (10 ,20, 35, 50, 75, 100): convert("bcms", "kmeans", k=k, header=(k==10))
	for k in (10 ,20, 35, 50, 75, 100): convert("de-at", "kmeans", k=k, header=(k==10))


def grid_miikka():
	# s represents the cell size in lat/lon degrees
	for s in (0.1, 0.2, 0.5): convert("ch", "miikka", cellsizeX=s, cellsizeY=s, header=(s==0.1))
	for s in (0.5, 1, 2): convert("bcms", "miikka", cellsizeX=s, cellsizeY=s, header=(s==0.5))
	for s in (0.5, 1, 2): convert("de-at", "miikka", cellsizeX=s, cellsizeY=s, header=(s==0.5))


def grid_tommi():
	# s represents the number of parts the grid is divided into (horizontally and vertically)
	s = 9
	convert("ch", "tommi", numcellsX=s, numcellsY=s, header=True)
	convert("bcms", "tommi", numcellsX=s, numcellsY=s, header=True)
	convert("de-at", "tommi", numcellsX=s, numcellsY=s, header=True)


if __name__ == "__main__":
	kmeans()
	grid_miikka()
	grid_tommi()

