package de.ukoeln.idh.teaching.jml.ex05;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.Filter;

import java.util.HashMap;
import java.util.Arrays;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

public class Main {

	public static void main(String[] args) throws Exception {
		// code goes here :)

		//init
		DataSource source = new DataSource("src/main/resources/germancredit/train.arff");
    	Instances data = source.getDataSet();
    	if (data.classIndex() == -1) {
      		data.setClassIndex(data.numAttributes() - 1);
		}
		System.out.println("N Instances: "+data.size());
		//split
		RemovePercentage rp = new RemovePercentage();
		rp.setPercentage(90);
		rp.setInputFormat(data);
		Instances devSet = Filter.useFilter(data, rp);
		rp.setInvertSelection(true);
		rp.setInputFormat(data);
		Instances trainSet = Filter.useFilter(data, rp);

		System.out.println("Dev: "+devSet.size()+" Train: "+trainSet.size());

		//params to test
		int[] numIter = new int[] {5, 10, 25, 50, 100, 200};
		int[] treeDepth = new int[] {0, 5, 10, 25, 50, 100, 200};
		int[] numFeat = new int[] {5, 10, 15, 20};
		boolean[] breakTies = new boolean[] {true, false};

		Map<int[], Double> results = new HashMap<>();
		
		//init RF
		RandomForest rf = new RandomForest();
		Random randy = new Random(1);

		double bestFScore = Double.MIN_VALUE;
		int[] bestParam = null;
		String[] options = null;

		for(int iter:numIter){
			for(int depth:treeDepth){
				for(int feat:numFeat){
					for(boolean breakTie:breakTies){
						rf.setNumIterations(iter);
						rf.setMaxDepth(depth);
						rf.setNumFeatures(feat);
						rf.setBreakTiesRandomly(breakTie);

						Evaluation eval = new Evaluation(devSet);
						eval.crossValidateModel(rf, devSet, 5, randy);

						double fScore = eval.weightedFMeasure();

						results.put(new int[] {iter, depth, feat, (breakTie ? 1:0)}, fScore);

						if(fScore > bestFScore){
							bestFScore = fScore;
							bestParam = new int[] {iter, depth, feat, (breakTie ? 1:0)};
							//Try getting options for easier implementation of training forest
							options = rf.getOptions();
						}
					}
				}
			}
		}

		System.out.println("Best Param Config: " + Arrays.toString(bestParam));
		System.out.println("Best Param Options: "+ Arrays.asList(options));

		//build with best-performing params

		RandomForest rfTrain = new RandomForest();
		rfTrain.setOptions(options);
		rfTrain.buildClassifier(trainSet);

		DataSource testSource = new DataSource("src/main/resources/germancredit/test.arff");
    	Instances testData = testSource.getDataSet();
    	if (testData.classIndex() == -1) {
      		testData.setClassIndex(data.numAttributes() - 1);
		}

		Evaluation testEval = new Evaluation(testData);
		testEval.evaluateModel(rfTrain, testData);
		System.out.println(testEval.toClassDetailsString());

		


	}

	
}