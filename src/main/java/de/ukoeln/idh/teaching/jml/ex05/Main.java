package de.ukoeln.idh.teaching.jml.ex05;
import java.io.File;
import java.io.IOException;

import java.util.TreeMap;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

/*results: best f1score on devset: ~0,75
 params: 100 iterations, 10 depth, 5 features 
 f1score on testset: ~0,64*/

public class Main {

	public static void main(String[] args) throws Exception {
		// load data
		Instances instances = loadFile("src/main/resources/germancredit/train.arff");
		Instances testSet = loadFile("src/main/resources/germancredit/test.arff");
		testSet.setClassIndex(testSet.numAttributes()-1);
		
		
		// split instances in train and dev
		//train
		RemovePercentage rmperc = new RemovePercentage();
		rmperc.setPercentage(90);
		rmperc.setInputFormat(instances);
		Instances trainSet = Filter.useFilter(instances, rmperc);
		trainSet.setClassIndex(trainSet.numAttributes()-1);
		
		//dev
		rmperc.setInvertSelection(true);
		rmperc.setInputFormat(instances);
		Instances devSet = Filter.useFilter(instances, rmperc);
		devSet.setClassIndex(devSet.numAttributes()-1);
		
		
		//parameters
		
		//numIterations
		int[] iterations = {50, 100, 150, 200, 250, 300, 350, 400};
		//batchsize
		//String[] batchsize = {"50", "100", "150", "200"};
		//maxDepth
		int[] depth = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60};
		//NumFeatures
		int[] featurenumber = {5, 10, 15, 20};
		
		//number of models
		int NumModels = iterations.length * depth.length * featurenumber.length;
		System.out.println(NumModels + " models will be trained.");
		
		//for saving scores (TreeMap for sorted order of keys)
		TreeMap<Double, int[]> scores = new TreeMap<Double, int[]>(Collections.reverseOrder());
		
		int index = 1;
		//iterate over parameters
		for(int i:iterations) {
			//for(String b:batchsize) {
			for(int d:depth) {
				for(int fn:featurenumber) {
					RandomForest rf = new RandomForest();
					rf.setSeed(0);
					rf.setNumIterations(i);
					rf.setMaxDepth(d);
					rf.setNumFeatures(fn);
					Evaluation eval = new Evaluation(devSet);
					eval.crossValidateModel(rf, devSet, 10, new Random(0));
						
					//save fscore
					int[] params =  {i, d, fn};
					scores.put(eval.weightedFMeasure(), params);
					
					System.out.println("trained model number "+ index);
					index ++;
				}
			}
		}
		//get best fscore of models on devset
		double bestScore = scores.firstKey();
		int[] bestParams = scores.get(bestScore);
		System.out.println("best params on devset: " + Arrays.toString(bestParams));
		System.out.println("best score on devset:" + bestScore);
		
		//train on trainset
		System.out.println("now training on trainset...");
		RandomForest rf = new RandomForest();
		rf.setSeed(0);
		rf.setNumIterations(bestParams[0]);
		rf.setMaxDepth(bestParams[1]);
		rf.setNumFeatures(bestParams[2]);
		rf.buildClassifier(trainSet);
		
		//evaluate on testset
		Evaluation eval = new Evaluation(testSet);
		eval.evaluateModel(rf, testSet);
		System.out.println(eval.toSummaryString());
		System.out.println(eval.weightedFMeasure());
		
	}
	
	private static Instances loadFile(String pathToFile) throws IOException {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(pathToFile));
		Instances instances = loader.getDataSet();
		return instances;
	}

	
}
	