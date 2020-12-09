package de.ukoeln.idh.teaching.jml.ex05;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;


public class Main {

	public static void main(String[] args) throws Exception {
		
		BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/amazon/train.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances wholeTrainingData = arff.getData();
		wholeTrainingData.setClassIndex(wholeTrainingData.numAttributes()-1);
		
		//Trainingsdaten splitten
		
		int devision = (int)(0.1d * wholeTrainingData.size());
		int trainSize = wholeTrainingData.size() - devision;
		
		Instances devData = new Instances(wholeTrainingData, 0, devision);
		Instances trainingData = new Instances(wholeTrainingData, devision, trainSize);
		
		System.out.println("Test Data: " + devData.size() + "\tTraining Data: " + trainingData.size());
		

		Integer[] depthArray = new Integer[] { 2, 8, 1 };
		Integer[] iterations = new Integer[] { 50, 300, 50};
		Integer[] numFeatures = new Integer[] { 2, 20, 1 };
		boolean[] breakTiesRandom = new boolean[] { false, true };

		Map<Integer[], Double> results = new HashMap<>();

		//beste Konfiguration und besten F-Score speichern
		Integer[] bestConfig = null; 
		double bestFScore = Double.MIN_VALUE; 
		int i = 0;
		RandomForest classifier = new RandomForest();
		for (int depth = depthArray[0]; depth <= depthArray[1]; depth += depthArray[2]) {
			classifier.setMaxDepth(depth);
			for (int iter = iterations[0]; iter <= iterations[1]; iter += iterations[2]) {
				classifier.setNumIterations(iter);
				for (int numF = numFeatures[0]; numF <= numFeatures[1]; numF += numFeatures[2]) {
					classifier.setNumFeatures(numF);
					for (boolean breakRandom : breakTiesRandom) {
						classifier.setBreakTiesRandomly(breakRandom);
						i++;
						if ((i % 500) == 0)
							System.out.println(i + " configurations crossvalidated");

						Evaluation eval = new Evaluation(devData);
						eval.crossValidateModel(classifier, devData, 10, new Random(15));
						double f = eval.weightedFMeasure();
						results.put(new Integer[] { depth, iter, numF, (breakRandom ? 1 : 0) }, f);
						if (f > bestFScore) {
							bestFScore = f;
							bestConfig = new Integer[] { depth, iter, numF, (breakRandom ? 1 : 0) };
						}

					}

				}
			}
		}

		// alle Ergebnisse mit F-Score sortiert ausgeben
		LinkedHashMap<Integer[], Double> reverseSortedMap = new LinkedHashMap<>();
		results.entrySet().stream().sorted(Map.Entry.comparingByValue())
				.forEachOrdered(x -> reverseSortedMap.put(x.getKey(), x.getValue()));
		for (Map.Entry<Integer[], Double> e : reverseSortedMap.entrySet()) {
			System.out.println(
					"[Depth, MaxIter, MaxFeatures, BreakRandom]=" + Arrays.asList(e.getKey()) + " --> " + e.getValue());
		}

		System.out.println("Best Parameter Configuration: " + Arrays.asList(bestConfig));

		// mit bester Konfiguration neues Modell mit Trainingsdaten erstellen
		classifier = new RandomForest();
		classifier.setMaxDepth(bestConfig[0]);
		classifier.setNumIterations(bestConfig[1]);
		classifier.setNumFeatures(bestConfig[2]);
		classifier.setBreakTiesRandomly(bestConfig[3] == 1);

		classifier.buildClassifier(trainingData);

		// Testdaten einlesen und Modell anwenden
		reader = new BufferedReader(new FileReader("src/main/resources/amazon/test.arff"));
		arff = new ArffReader(reader);
		Instances testData = arff.getData();
		testData.setClassIndex(testData.numAttributes() - 1);

		Evaluation eval = new Evaluation(testData);
		eval.evaluateModel(classifier, testData);
		System.out.println(eval.toClassDetailsString());
		
	}

	
}