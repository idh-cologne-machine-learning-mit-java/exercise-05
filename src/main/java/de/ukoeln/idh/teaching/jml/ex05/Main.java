package de.ukoeln.idh.teaching.jml.ex05;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import weka.filters.Filter;
import weka.core.Utils;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import weka.filters.unsupervised.instance.RemovePercentage;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;


public class Main {

	public static void main(String[] args) throws Exception {
		// load data
		Instances instances = loadData("src/main/resources/amazon/train.arff.gz");
		instances.setClassIndex(instances.numAttributes() - 1);
		Instances test_instances = loadData("src/main/resources/amazon/test.arff.gz");
		test_instances.setClassIndex(test_instances.numAttributes() - 1);
		
		// split training data set into training and tuning data
		RemovePercentage rm = new RemovePercentage();
		rm.setPercentage(90);
		rm.setInputFormat(instances);
		Instances tuning_instances = Filter.useFilter(instances, rm);
		rm.setInvertSelection(true);
		rm.setInputFormat(instances);
		Instances training_instances = Filter.useFilter(instances, rm);

		//
		// parameter selection
		//
		RandomForest rf = new RandomForest();
		rf.setSeed(1);
		rf.setNumExecutionSlots(0); // auto-detect number of cores

		// generate parameter matrix
		ImmutableList possibleParameters = ImmutableList.of(
				ImmutableSet.of(1, 50, 100, 150, 200, 250, 300, 350, 400), // number of trees parameter
			    ImmutableSet.of(0, 10, 25, 50, 100, 150, 200), // max depth
			    ImmutableSet.of(1, 2, 4)); // minimum number of instances per leaf
		
		Set parameterMatrix = Sets.cartesianProduct(possibleParameters);

		
		// test parameters
		double bestF = 0;
		String[] bestOptions = {""};
		
		Iterator<ImmutableList> it = parameterMatrix.iterator();
	    while(it.hasNext()){
	    	// set current test parameters
	    	ImmutableList currentParameters = it.next();
	    	rf.setOptions(new String[]{
	    			"-I", currentParameters.get(0).toString(),
	    			"-depth", currentParameters.get(1).toString(),
	    			"-M", currentParameters.get(2).toString()
	    	});
	    	System.out.print("Testing paramerers: ");
	    	System.out.print(Utils.joinOptions(rf.getOptions()));
	    	
	    	// evaluate with current parameters
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
		
	    //
	    // training
	    //
	    RandomForest bestRf = new RandomForest();
	    bestRf.setOptions(bestOptions);
	    bestRf.buildClassifier(training_instances);
	    
	    //
	    // evaluation
	    //
	    Evaluation eval = new Evaluation(test_instances);
	    eval.evaluateModel(bestRf, test_instances);
	    
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
	}
	
	public static Instances loadData(String path) throws IOException {
		  ArffLoader loader = new ArffLoader();
		  loader.setFile(new File(path));
		  return loader.getDataSet();
	}

	
}