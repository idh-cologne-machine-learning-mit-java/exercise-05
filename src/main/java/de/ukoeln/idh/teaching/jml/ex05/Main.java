package de.ukoeln.idh.teaching.jml.ex05;

import java.io.File;
import java.util.Random;
import java.util.Map;
import java.util.HashMap;

import weka.core.converters.ArffLoader;
import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;


public class Main {

	public static void main(String[] args) throws Exception {
		// load training data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("src/main/resources/germancredit/train.arff"));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// split training data (90% training - 10% dev)
		int trainCompleteDataSize = instances.size();
		int trainDataSize = (int)(trainCompleteDataSize * 0.9);
		int devDataSize = (int)(trainCompleteDataSize - trainDataSize);

		Instances devData = new Instances(instances, 0, devDataSize);
		Instances trainData = new Instances(instances, devDataSize, trainDataSize);

		// create hyperparameter sets
		int[] numTrees = {10, 50, 100};
		int[] depthTrees = {10, 50, 100};
		int[] numFeatures = {5, 10, 20};
		boolean[] breakTies = {true, false};

		// test hyperparameters on devData
		RandomForest classifier = new RandomForest();
		int seed = 42;
		Random random = new Random(seed);
		Map<int[], Double> results = new HashMap<>();
		double bestEval = 0;
		int[] bestHp = null;
		int intTies = 1;

		for (int trees:numTrees) {
			classifier.setNumIterations(trees);
			for (int depth:depthTrees) {
				classifier.setMaxDepth(depth);
				for (int features:numFeatures) {
					classifier.setNumFeatures(features);
					for (boolean ties:breakTies) {
						classifier.setBreakTiesRandomly(ties);

						Evaluation evaluation = new Evaluation(devData);
						evaluation.crossValidateModel(classifier, devData, 10, random);
						double eval = evaluation.weightedFMeasure();

						if (!ties) {
							intTies = 0;
						}

						results.put(new int[]{trees, depth, features, intTies}, eval);
						if (eval > bestEval) {
							bestEval = eval;
							bestHp = new int[]{depth, trees, features, intTies};
						}
					}
				}
			}
		}

		// convert breakTies hp to boolean (again)
		boolean trainTies = true;
		if (bestHp[3] == 0) {
			trainTies = false;
		}

		// build classifier on training data with best hyperparameters
		classifier.setNumIterations(bestHp[0]);
		classifier.setMaxDepth(bestHp[1]);
		classifier.setNumFeatures(bestHp[2]);
		classifier.setBreakTiesRandomly(trainTies);
		classifier.buildClassifier(trainData);

		// load test data
		loader.setFile(new File("src/main/resources/germancredit/test.arff"));
		Instances testInstances = loader.getDataSet();
		testInstances.setClassIndex(testInstances.numAttributes() - 1);

		// evaluate classifier on test data
		Evaluation evaluation = new Evaluation(testInstances);
		evaluation.evaluateModel(classifier, testInstances);
		System.out.println(evaluation.toClassDetailsString());

	}

}