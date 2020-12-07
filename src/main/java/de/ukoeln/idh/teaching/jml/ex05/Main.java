package de.ukoeln.idh.teaching.jml.ex05;


import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {
	public static void main(String[] args) throws Exception {
		// code goes here :)

		Instances tuningData;
		Instances trainData;
		Instances testData;

		CredibilityAnalyzer ca = new CredibilityAnalyzer();
		Instances[] data;

		data = ca.splitData(ca.load("src/main/resources/germancredit/train.arff"));

		/** verstehe nicht warum das die gleiche groesse hat. */
		System.out.println(data[0].size());
		System.out.println(data[1].size());

		tuningData = data[0];
		tuningData.setClassIndex(tuningData.numAttributes() - 1);
		trainData = data[1];
		trainData.setClassIndex(tuningData.numAttributes() - 1);

		/** possible values - no special reason why i have chosen them except for num features as max is 20 **/
		int[] numIterations = {1, 2, 5, 10, 20, 50};
		int[] batchSizes = {1, 2, 5, 20, 50, 100};
		int[] maxDepths = { 1, 2, 5, 20, 50, 100};
		int[] numFeatures = {1, 2, 3, 4, 5, 10, 15, 20};
		RandomForest classifier = new RandomForest();
		classifier.setNumExecutionSlots(0);
		double bestF = 0;
		int bestNumIteration = 1;
		int bestBatchSize = 1;
		int bestMaxDepth = 1;
		int bestNumFeature = 1;
		for (int numIteration : numIterations) {
			for (int batchSize : batchSizes) {
				for (int maxDepth : maxDepths) {
					for (int numFeature : numFeatures) {
						classifier.setNumIterations(numIteration);
						classifier.setBatchSize(batchSize + "");
						classifier.setMaxDepth(maxDepth);
						classifier.setNumFeatures(numFeature);
						double nextF = ca.getF(tuningData, classifier);
						if (nextF > bestF) {
							bestF = nextF;
							bestNumIteration = numIteration;
							bestBatchSize = batchSize;
							bestMaxDepth = maxDepth;
							bestNumFeature = numFeature;
						}
					}
				}
			}
		}

		System.out.println("Best F is " + bestF + "with Parameters:");
		System.out.println("numIteration: " + bestNumIteration);
		System.out.println("bestBatchSize: " + bestBatchSize);
		System.out.println("bestMaxDepth: " + bestMaxDepth);
		System.out.println("bestNumFeature: " + bestNumFeature);
		/** train **/


		classifier.setNumIterations(bestNumIteration);
		classifier.setBatchSize(bestBatchSize + "");
		classifier.setMaxDepth(bestMaxDepth);
		classifier.setNumFeatures(bestNumFeature);

		classifier.buildClassifier(trainData);

		testData = ca.load("src/main/resources/germancredit/train.arff");
		testData.setClassIndex(testData.numAttributes() - 1);

		Evaluation e = new Evaluation(testData);

		e.evaluateModel(classifier, testData);

		System.out.println(e.toClassDetailsString());

	}

}