package de.ukoeln.idh.teaching.jml.ex05;


import java.util.Random;
import java.util.TreeMap;

import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;


public class Main {

	//This main function is to long and should be split into sub-classes and functions later.
	public static void main(String[] args) throws Exception {
		DataSource loader = new DataSource("src/main/resources/germancredit/train.arff");
		Instances data = loader.getDataSet();
		
		int seed = 1;
		Random r = new Random(seed);
		Instances[] splits = splitDataset(data, 0.9, r);
		Instances trainSet = splits[0];
		Instances devSet = splits[1];
		
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		devSet.setClassIndex(devSet.numAttributes() - 1);
		
		int[] numTrees = {10, 25, 50, 100, 200};
		String[] batchSizes = {"50", "100", "200"};
		int[] treeDepth = {0, 10, 25, 50, 100, 200};
		int[] numFeatures = {0, 5, 10, 15, 20};
		boolean[] breakTiesRandomly = {true, false };
		
		int numModels = numTrees.length * batchSizes.length * treeDepth.length * numFeatures.length * breakTiesRandomly.length;
		System.out.println("Performing hyperparameter testing on " + numModels + " Models.");
		
		TreeMap<Double, ModelScore> scores = new TreeMap<Double, ModelScore>();
		
		int index = 0;
		for(int n:numTrees) {
			for(String b:batchSizes) {
				for(int d: treeDepth) {
					for(int nf: numFeatures) {
						for(boolean t: breakTiesRandomly) {
							RandomForest forest = new RandomForest();
							forest.setNumIterations(n);
							forest.setBatchSize(b);
							forest.setMaxDepth(d);
							forest.setNumFeatures(nf);
							forest.setBreakTiesRandomly(t);
							Evaluation eval = new Evaluation(devSet);
							eval.crossValidateModel(forest, devSet, 10, r);
							
							ModelScore score = new ModelScore(eval.weightedFMeasure(), n, b, d, nf, t);	
							scores.put(eval.weightedFMeasure(), score);
							
							index++;
							if(((double)index / numModels * 100) % 10 == 0) {
								System.out.println("Evaluated model " + index + " (" + (double)index / numModels * 100 + "% of all models).");
							}
						}
					}
				}
			}
		}
		ModelScore model = scores.get(scores.descendingKeySet().first());
		System.out.println("");
		System.out.println("Best performing model:");
		model.printModel();
		System.out.println("");
		
		System.out.println("Training classifier.");
		RandomForest forest = new RandomForest();
		forest.setNumIterations(model.numTrees);
		forest.setBatchSize(model.batchSize);
		forest.setMaxDepth(model.treeDepth);
		forest.setNumFeatures(model.numFeatures);
		forest.setBreakTiesRandomly(model.breakTiesRandomly);
		forest.buildClassifier(trainSet);
		
		DataSource testLoader = new DataSource("src/main/resources/germancredit/test.arff");
		Instances testSet = testLoader.getDataSet();
		testSet.setClassIndex(testSet.numAttributes() - 1);
		
		System.out.println("Testing classifier.");
		Evaluation eval = new Evaluation(testSet);
		eval.evaluateModel(forest, testSet);
		System.out.println(eval.toSummaryString());
		System.out.println("Weighted F1 score: " + eval.weightedFMeasure());
	}
	
	public static Instances[] splitDataset(Instances dataset, double split) {
		Instances split1, split2;
		int instancesInSplit2 = (int)Math.ceil((1 - split) * dataset.numInstances());
		
		split1 = new Instances(dataset, instancesInSplit2, dataset.numInstances() - instancesInSplit2);
		split2 = new Instances(dataset, 0, instancesInSplit2);
		
		Instances[] splits = new Instances[2];
		splits[0] = split1;
		splits[1] = split2;
		
		return splits;
	}
	
	
	public static Instances[] splitDataset(Instances dataset, double split, Random random) {
		Instances split1, split2;
		int instancesInSplit2 = (int)Math.ceil((1 - split) * dataset.numInstances());
		
		dataset.randomize(random);
		split1 = new Instances(dataset, instancesInSplit2, dataset.numInstances() - instancesInSplit2);
		split2 = new Instances(dataset, 0, instancesInSplit2);
		
		Instances[] splits = new Instances[2];
		splits[0] = split1;
		splits[1] = split2;
		
		return splits;
	}

	
}