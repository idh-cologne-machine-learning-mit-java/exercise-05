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

	public static void main(String[] args) throws Exception {
		// Trainingsdaten einlesen
		BufferedReader reader = new BufferedReader(new FileReader("src/main/resources/germancredit/train.arff"));
		ArffReader arff = new ArffReader(reader);
		Instances allTrainData = arff.getData();
		allTrainData.setClassIndex(allTrainData.numAttributes() - 1);

		// 10 % der Trainingsdaten für Hyperparameter Tuning nutzen, übrige 90% für
		// anschließendes Training speichern
		int devSize = (int) (0.1d * allTrainData.size());
		int trainSize = allTrainData.size() - devSize;

		Instances devData = new Instances(allTrainData, 0, devSize); // 10 %
		Instances trainData = new Instances(allTrainData, devSize, trainSize); // 90%
		System.out.println("Dev Data: " + devData.size() + "\tTrainData: " + trainData.size());

		// configuration arrays: index0=minimun value; index1=maximum value; index2=step
		// size
		// z.B. new Integer[] {2, 8, 1} erzeugt Bäume mit depth = 2,3,4,5,6,7,8
		Integer[] depthArray = new Integer[] { 2, 8, 1 };
		Integer[] iterations = new Integer[] { 50, 300, 50};
		Integer[] numFeatures = new Integer[] { 2, 20, 1 };
		boolean[] breakTiesRandom = new boolean[] { false, true };

		Map<Integer[], Double> results = new HashMap<>();

		Integer[] bestConfig = null; // speichert beste Konfiguration
		double bestFScore = Double.MIN_VALUE; // speichert höchsten F-Score
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

		// alle Ergebnisse mit F-Score nach F-Score sortiert ausgeben
		LinkedHashMap<Integer[], Double> reverseSortedMap = new LinkedHashMap<>();
		results.entrySet().stream().sorted(Map.Entry.comparingByValue())
				.forEachOrdered(x -> reverseSortedMap.put(x.getKey(), x.getValue()));
		for (Map.Entry<Integer[], Double> e : reverseSortedMap.entrySet()) {
			System.out.println(
					"[Depth, MaxIter, MaxFeatures, BreakRandom]=" + Arrays.asList(e.getKey()) + " --> " + e.getValue());
		}

		System.out.println("Best Config: " + Arrays.asList(bestConfig));

		// mit bester Konfiguration neues Modell mit Trainingsdaten erstellen
		classifier = new RandomForest();
		classifier.setMaxDepth(bestConfig[0]);
		classifier.setNumIterations(bestConfig[1]);
		classifier.setNumFeatures(bestConfig[2]);
		classifier.setBreakTiesRandomly(bestConfig[3] == 1);

		classifier.buildClassifier(trainData);

		// Testdaten einlesen und Modell evaluieren
		reader = new BufferedReader(new FileReader("src/main/resources/germancredit/test.arff"));
		arff = new ArffReader(reader);
		Instances testData = arff.getData();
		testData.setClassIndex(testData.numAttributes() - 1);

		Evaluation eval = new Evaluation(testData);
		eval.evaluateModel(classifier, testData);
		System.out.println(eval.toClassDetailsString());

	}

}