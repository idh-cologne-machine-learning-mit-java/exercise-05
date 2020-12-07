package de.ukoeln.idh.teaching.jml.ex05;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {

	public static void main(String[] args) throws Exception {
		// code goes here :)
		
		Instances instances = loadData("src/main/resources/amazon/train.arff.gz");
		instances.setClassIndex(instances.numAttributes() - 1);
		Instances test_instances = loadData("src/main/resources/amazon/test.arff.gz");
		test_instances.setClassIndex(test_instances.numAttributes() - 1);

		// Split des Datensatzes in Trainings und Tuning Daten
		RemovePercentage rm = new RemovePercentage();
		rm.setPercentage(90);
		rm.setInputFormat(instances);
		Instances tuning_instances = Filter.useFilter(instances, rm);
		rm.setInvertSelection(true);
		rm.setInputFormat(instances);
		Instances training_instances = Filter.useFilter(instances, rm);

		
		// Parameterwahl
		
		RandomForest rf = new RandomForest();
		rf.setSeed(1);
		rf.setNumExecutionSlots(0); // auto-detect number of cores

		// generate parameter matrix
		ImmutableList parameters = ImmutableList.of(ImmutableSet.of(1, 50, 100, 150, 200, 250, 300, 350, 400), // Decision Tree Zahlen 
																														
				ImmutableSet.of(0, 10, 25, 50, 100, 150, 200), 
				ImmutableSet.of(1, 2, 4)); 

		Set parameterMatrix = Sets.cartesianProduct(parameters);

		// Test-Parameter
		double bestF = 0;
		String[] bestOptions = { "" };

		Iterator<ImmutableList> it = parameterMatrix.iterator();
		while (it.hasNext()) {
			
			ImmutableList currentParameters = it.next();
			rf.setOptions(new String[] { "-I", currentParameters.get(0).toString(), "-depth",
					currentParameters.get(1).toString(), "-M", currentParameters.get(2).toString() });
			System.out.print("Testing paramerers: ");
			System.out.print(Utils.joinOptions(rf.getOptions()));

			// Evaluation
			Evaluation parameterTestEval = new Evaluation(tuning_instances);
			parameterTestEval.crossValidateModel(rf, tuning_instances, 10, new Random(1));

			double f = parameterTestEval.weightedFMeasure();
			System.out.println("    weighted fMeasure: " + f);
			if (f > bestF) {
				bestF = f;
				bestOptions = rf.getOptions();
			}
		}

		System.out.println("Best options: " + Utils.joinOptions(bestOptions) + " (fMeasure: " + bestF + ")");

		//Training
	
		RandomForest bestRandomForest = new RandomForest();
		bestRandomForest.setOptions(bestOptions);
		bestRandomForest.buildClassifier(training_instances);

	
		// Evaluation
		
		Evaluation evalu = new Evaluation(test_instances);
		evalu.evaluateModel(bestRandomForest, test_instances);

		System.out.println(evalu.toSummaryString());
		System.out.println(evalu.toMatrixString());
	}

	public static Instances loadData(String path) throws IOException {
		ArffLoader arffloader = new ArffLoader();
		arffloader.setFile(new File(path));
		return arffloader.getDataSet();
	}
}
