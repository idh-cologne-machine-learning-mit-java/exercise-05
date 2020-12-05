package de.ukoeln.idh.teaching.jml.ex05;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeMap;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {

	public static void main(String[] args) throws Exception {
	
		 Instances importedData = new Instances(new BufferedReader(new FileReader("src/main/resources/germancredit/train.arff")));
		 Instances importedTestData = new Instances(new BufferedReader(new FileReader("src/main/resources/germancredit/test.arff")));
		
		 
		 if (importedData.classIndex() == -1)
			 importedData.setClassIndex(importedData.numAttributes() - 1);
		 
		 if (importedTestData.classIndex() == -1)
			 importedTestData.setClassIndex(importedData.numAttributes() - 1);
		

		 Instances trainingData = dataSplitter(importedData, 90.0, true);
		 Instances paraTuningData = dataSplitter(importedData, 90.0, false);
		 
	
		 
		//Save Data for later
		
		saver(trainingData, "src/main/resources/germancredit/traindump.arff"); 
		saver(paraTuningData, "src/main/resources/germancredit/paradump.arff");
		 
		
		
		RandomForest rf = new RandomForest();
		rf.setNumExecutionSlots(0);
		
//		rf.setBagSizePercent(50);
//		rf.setBatchSize("100");
//		rf.setBreakTiesRandomly(true);
//		rf.setMaxDepth(1);
//		rf.setNumDecimalPlaces(2);
//		rf.setNumExecutionSlots(12);
//		rf.setNumFeatures(5);
//		rf.setNumIterations(100);
//		rf.setSeed(1);

		
		

		
		int[] numIterations = {1, 10, 20, 30, 40, 50}; //+30 macht scheinbar keinen Unterschied mehr
		int[] batchSizes = {1, 50, 100, 200, 1000};
		int[] maxDepth = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; //+30 macht scheinbar keinen Unterschied mehr
		int[] numFeatures = {1, 2, 3, 4, 5, 10, 15, 20};
		
		
		boolean[] attributeImportance = {true, false};
		boolean[] breakTies = {true, false};
		
		int computeLength = numIterations.length * batchSizes.length * maxDepth.length * numFeatures.length * attributeImportance.length * breakTies.length;
		
		
		int counter = 0;
		int seed = 1;
		Random randomSeed = new Random(seed);
		TreeMap<Double, Model> results = new TreeMap<Double, Model>();
		
		
		
		for(int i: numIterations) {
			rf.setNumIterations(i);
			for(int j: batchSizes) {
				rf.setBatchSize(Integer.toString(j));
				for(int k: maxDepth) {
					rf.setMaxDepth(k);
					for(int m: numFeatures) {
						rf.setNumFeatures(m);
						for(boolean t: attributeImportance) {
							rf.setComputeAttributeImportance(t);
							for(boolean z: breakTies) {
								rf.setBreakTiesRandomly(z);
								
								Evaluation eval = new Evaluation(paraTuningData);
								eval.crossValidateModel(rf, paraTuningData, 10, randomSeed);
								
								double score = eval.weightedFMeasure();
								
								
								if(score > 0) {
									Model settings = new Model(i, j , k, m, t, z, score);
									
									
									results.put(score, settings);
									
									counter++;
									System.out.println(counter + " von insgesamt " + computeLength + " Modellen fertig gestellt mit einem F-Score von " + score);
								}
							
							}
						}
							
					}
				}
			}
		}
		
		// old 
		
//		for(int i = 1; i <= 100; i+=10) {
//			rf.setNumIterations(i);
//			for(int j = 1; j <= 100; j+=10 ) {
//				rf.setBatchSize(Integer.toString(j));
//				for(int k = 1; k <= 100; k+=10) {
//					rf.setMaxDepth(k);
//					for(int m = 1; m <= 20; m+=15) {
//						rf.setNumFeatures(m);
//						
//						Evaluation eval = new Evaluation(paraTuningData);
//						eval.crossValidateModel(rf, paraTuningData, 10, randomSeed);
//						
//						double score = eval.weightedFMeasure();
//						
//						Model settings = new Model(i, j , k, m, score);
//						
//	
//						results.put(score, settings);
//						
//						counter++;
//						System.out.println(counter + " von insgesamt 4000 Modellen fertig gestellt mit einem F-Score von " + score);
//									
//					}
//				}
//			}	
//		}
		
	
	
	Model bestModel = results.get(results.descendingKeySet().first());
	System.out.println("Bestes Modell");
	bestModel.printModel();
	
	RandomForest finalResult = new RandomForest();
	finalResult.setNumIterations(bestModel.numFeatures);
	finalResult.setBatchSize(Integer.toString(bestModel.batchSize));
	finalResult.setMaxDepth(bestModel.maxDepth);
	finalResult.setNumFeatures(bestModel.numFeatures);
	finalResult.buildClassifier(trainingData);
	
	System.out.println("--------- Endergebnis --------");
	
	Evaluation eval = new Evaluation(importedTestData);
	
	eval.evaluateModel(finalResult, importedTestData);
	
	System.out.println(eval.toClassDetailsString());
	
		
		 
	}
	
	public static Instances dataSplitter(Instances inputData, double percent, boolean invert) throws Exception {
		
		RemovePercentage split = new RemovePercentage();
		split.setInvertSelection(invert);
		split.setPercentage(percent);
		split.setInputFormat(inputData);
		
		Instances outputData = new Instances(Filter.useFilter(inputData, split));
		
		return outputData; 
	}
	
	public static void saver(Instances data, String savepath) {
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		try {
			saver.setFile(new File(savepath));
			saver.writeBatch();
		} catch (IOException e) {
			
			e.printStackTrace();
		}
		
		
	}
	
}