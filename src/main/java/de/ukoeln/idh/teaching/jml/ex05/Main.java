package de.ukoeln.idh.teaching.jml.ex05;

import java.io.File;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Main {
	
	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();		
		
		loader.setFile(getFile("train"));
		Instances trainData = loader.getDataSet();
		
		trainData.setClassIndex(trainData.numAttributes() - 1);

		// 1. -----------------------------------------------------
		
		int tuneToCopy = trainData.size() / 10;
		int tuneFirst = trainData.size() - tuneToCopy;
	
		Instances tuneData = new Instances(trainData, tuneFirst, tuneToCopy);	// last 10 %
		trainData = new Instances(trainData, 0, tuneFirst);						// first 90 %
		
		// 2. -----------------------------------------------------

		RandomForest classifier = null;
		double f = 0;
		
		for (int i = 0; i < 10; i++) {
			RandomForest curClassifier = new RandomForest();
			curClassifier.setMaxDepth(i);
			
			for (int j = 1; j <= 50; j+=5) {
				curClassifier.setNumIterations(j);
			
				for (int k = 0; k < tuneData.numAttributes(); k++) {
					// loops: etc...
					
					curClassifier.setNumFeatures(k);
					
					Evaluation evaluation = new Evaluation(tuneData);
					evaluation.crossValidateModel(curClassifier, tuneData, 10, new Random(1));
					
					double curF = evaluation.weightedFMeasure();
					
					if (curF > f) {
						classifier = curClassifier;
						f = curF;
					}
				}
			}
		}
		
		// 3. -----------------------------------------------------
		
		classifier.buildClassifier(trainData);
		
		// 4. -----------------------------------------------------
		
		loader.setFile(getFile("test"));
		Instances testData = loader.getDataSet();
		
		testData.setClassIndex(trainData.classIndex());
		
		Evaluation evaluation = new Evaluation(testData);
		evaluation.evaluateModel(classifier, testData);
		
		System.out.println(evaluation.toClassDetailsString());
//		System.out.println(evaluation.weightedFMeasure());
	}
	
	private static File getFile(String name) {
		return new File("src/main/resources/germancredit", name + ".arff");
	}
	
}