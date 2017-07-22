import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm

data_to_train = []
values_of_data = []

training_set_size = 3000
hop_size = 30

classifier = svm.SVC(gamma=0.001, C=100)

#Add 99% of data in HASYv2 dataset to training set
with open('./HASYv2/hasy-data-labels.csv', 'r') as csvfile:
	csvread = csv.reader(csvfile, delimiter=',')

	x = 0
	for row in csvread:
		if (x > training_set_size):
			break
		if ((x%hop_size) != 0) and (x != 0):
			png_img = plt.imread('HASYv2/' + row[0])
			png_img_flat = png_img.ravel()
			data_to_train.append(png_img_flat)
			values_of_data.append(row[2])
			print ("Loading image " + str(x))
		x += 1

print ("Training machine now...")

classifier.fit(data_to_train, values_of_data)

print ("Done")

with open('./HASYv2/hasy-data-labels.csv', 'r') as csvfile:
	csvread = csv.reader(csvfile, delimiter=',')

	x = 0
	for row in csvread:
		if (x > training_set_size):
			break
		if ((x%hop_size) == 0) and (x != 0):
			png_img = plt.imread('HASYv2/' + row[0])
			png_img_flat = png_img.ravel()

			guess = classifier.predict(png_img_flat.reshape(1,-1))[0]
			if guess == row[2]:
				print("Match")
			else:
				print("Mismatch")
			#print (guess)
			#print (row[2])
		x += 1