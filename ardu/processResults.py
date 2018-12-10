from sklearn import metrics
from collections import defaultdict,OrderedDict
import pylab as plt
from matplotlib.pyplot import cm
import numpy as np


#This module produces graaphs!
def processResults(prediction,testY,filling,method):


	#saveGraphs(prediction,testY,filling,method)

	return analyzeResults(prediction,testY,filling,method)





#def saveGraphs(prediction,testY,filling,method):


def analyzeResults(prediction,testY,filling,method):

	return metrics.classification_report(testY, prediction, output_dict=True)


def generateGraphs(processedResults,fillmethods):

	labels = processedResults.keys()


	for currentFill in fillmethods:
		color=iter(cm.rainbow(np.linspace(0,1,6)))
		for label in labels:
			c=next(color)
			labelResultList = processedResults[label]
			n_componentsList = []
			f1List = []
			for labelResultDict in labelResultList:
				n_components = labelResultDict["n_components"]
				filling = labelResultDict["filling"]
				if filling != currentFill:
					continue
				n_componentsList.append(n_components)
				f1List.append(labelResultDict['weighted avg']['f1-score'])

			plt.figure(currentFill)

			plt.plot(n_componentsList, f1List,c=c,label= label)
			plt.xlabel("Number of Components used on " + currentFill + "dataset")
			plt.ylabel('Weight Average F1 Score')

	for filling in fillmethods:
		plt.figure(filling)


		handles, labels = plt.gca().get_legend_handles_labels()
		by_label = OrderedDict(zip(labels, handles))
		plt.legend(by_label.values(), by_label.keys())

	plt.show()


def generateGraphsSingle(processedResults,fillmethods):

	labels = processedResults.keys()

	for label in labels:

		color=iter(cm.rainbow(np.linspace(0,1,6)))
		for currentFill in fillmethods:
			c=next(color)
			labelResultList = processedResults[label]
			n_componentsList = []
			f1List = []
			for labelResultDict in labelResultList:
				n_components = labelResultDict["n_components"]
				filling = labelResultDict["filling"]
				if filling != currentFill:
					continue
				n_componentsList.append(n_components)
				f1List.append(labelResultDict['weighted avg']['f1-score'])

			plt.figure(1)

			plt.plot(n_componentsList, f1List,c=c,label= currentFill)
			plt.xlabel("Number of Components used")
			plt.ylabel('Weight Average F1 Score')




	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys())

	plt.show()
