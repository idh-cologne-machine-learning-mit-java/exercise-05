package de.ukoeln.idh.teaching.jml.ex05;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

public class Main {
	
	private static String trainingdatafile = "src/main/resources/germancredit/train.arff";

	public static void main(String[] args) throws Exception {


		// load trainingdata and prepare trainInstances
		BufferedReader reader = new BufferedReader(new FileReader(trainingdatafile));
		ArffReader arffreader = new ArffReader(reader);
		Instances trainInstances = arffreader.getData();
		trainInstances.setClassIndex(trainInstances.numAttributes() - 1);
		
		// splitting the training-data into 90/10 sets
		// as we need 10% for parameter-testing and rest 90% for training
		int testpart = (int) (0.1 * trainInstances.size());
		int trainingpart = trainInstances.size() - testpart;
		
		Instances testdata = new Instances(trainInstances, 0, testpart); // 10 %-part of the dataset
		Instances traindata = new Instances(trainInstances, testpart, trainingpart); // 90%-part of the dataset
		System.out.println("Size of Test-Data: " + testdata.size() + "\r\nSize of Training-Data: " + trainInstances.size());
		
		// use smaller one in order to parameter tuning and larger one for proper training
		// do the configuration-arrays
		Integer[] treeDepths = new Integer[] { 2, 5, 1 };
		Integer[] iterations = new Integer[] { 10, 100, 50};
		Integer[] features = new Integer[] { 5, 18, 1 };
		
		Map<Integer[], Double> results = new HashMap<>();
		
		Integer[] suitableSettings = null; 
		double highestFScore = Double.MIN_VALUE;
		int i = 0;
		RandomForest classifier = new RandomForest();
		
		// systematically test "all" possible hyper parameter settings, using cross validation on the development data set
		for (int depth = treeDepths[0]; depth <= treeDepths[1]; depth += treeDepths[2]) {
			classifier.setMaxDepth(depth);
			for (int iter = iterations[0]; iter <= iterations[1]; iter += iterations[2]) {
				classifier.setNumIterations(iter);
				for (int numF = features[0]; numF <= features[1]; numF += features[2]) {
					
					classifier.setNumFeatures(numF);
					
					classifier.setBreakTiesRandomly(true);
					i++;
					if ((i % 500) == 0)
						System.out.println(i + " configurations crossvalidated");

					Evaluation eval = new Evaluation(testdata);
					eval.crossValidateModel(classifier, testdata, 10, new Random(27));
					double f = eval.weightedFMeasure();
					results.put(new Integer[] { depth, iter, numF, (true ? 1 : 0) }, f);
					if (f > highestFScore) {
						highestFScore = f;
						suitableSettings = new Integer[] { depth, iter, numF, (true ? 1 : 0) };
					}
					

				}
			}
		}
		
		// sort all values by f-score
		LinkedHashMap<Integer[], Double> reverseSortedMap = new LinkedHashMap<>();
		results.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEachOrdered(x -> reverseSortedMap.put(x.getKey(), x.getValue()));
		
		// print out values from the sorted map
		for (Map.Entry<Integer[], Double> e : reverseSortedMap.entrySet()) {
			System.out.println(
					"[Depth, MaxIterations, MaxFeatures, BreakRandom]=" + Arrays.asList(e.getKey()) + " ===> " + e.getValue());
		}

		System.out.println("Best Configuration settings: " + Arrays.asList(suitableSettings));
		
		// train a model on the remaining training data set
		classifier = new RandomForest();
		
		classifier.setMaxDepth(suitableSettings[0]);
		classifier.setNumIterations(suitableSettings[1]);
		classifier.setNumFeatures(suitableSettings[2]);
		classifier.setBreakTiesRandomly(suitableSettings[3] == 1);
		
		classifier.buildClassifier(traindata);

		
		// apply and evaluate on a data set
		reader = new BufferedReader(new FileReader("src/main/resources/germancredit/test.arff"));
		arffreader = new ArffReader(reader);
		
		Instances testData = arffreader.getData();
		testData.setClassIndex(testData.numAttributes() - 1);

		Evaluation eval = new Evaluation(testData);
		eval.evaluateModel(classifier, testData);
		
		// now print out detailed matrix-table with values...
		System.out.println(eval.toClassDetailsString());
	}

	
}