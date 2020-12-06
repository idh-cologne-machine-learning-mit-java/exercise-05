package de.ukoeln.idh.teaching.jml.ex05;

//import javax.sql.DataSource;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import java.util.Map;
import java.util.HashMap;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;


public class Main {

	public static void main(String[] args) throws Exception {
		// splitting data
		DataSource source = new DataSource ("src/main/resources/germancredit/train.arff");
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		int seed = 1;
		Random random = new Random(seed);

		int tuning = (int) (0.1 * dataset.size());
		Instances tuningData = new Instances(dataset, 0, tuning);

		int train = dataset.size() - tuning;
		Instances trainData = new Instances(dataset, tuning, train);

		// testing hyperparameters
		int[] treeDepth = new int[] {0, 10, 50, 100, 200};
		int[] numTrees = new int[] {0, 10, 50, 100, 200};
		int[] numFeatures = new int[] {5, 10, 15, 20};

		Map<int[], Double> results = new HashMap<>();

		RandomForest rf = new RandomForest();

		int[] bestOptions = null;
		double bestF = 0;

		for (int depth:treeDepth) {
			rf.setMaxDepth(depth);
			for (int trees:numTrees) {
				rf.setNumIterations(trees);
				for (int features:numFeatures) {
					rf.setNumFeatures(features);		
					
					Evaluation evaluation = new Evaluation(tuningData);
					evaluation.crossValidateModel(rf, tuningData, 10, random);
	
					double f = evaluation.weightedFMeasure();
					results.put(new int[] {depth, trees, features}, f);
					if (f > bestF) {
						bestF = f;
						bestOptions = new int[] {depth, trees, features};

					}

				}

			}
		}

		// training data
		rf.setMaxDepth(bestOptions[0]);
		rf.setNumIterations(bestOptions[1]);
		rf.setNumFeatures(bestOptions[2]);
		rf.buildClassifier(trainData);

		//testing data
		DataSource source1 = new DataSource ("src/main/resources/germancredit/test.arff");
		Instances testData = source1.getDataSet();
		testData.setClassIndex(testData.numAttributes()-1);

		Evaluation evaluationTest = new Evaluation(testData);
		evaluationTest.evaluateModel(rf, testData);
		
		System.out.println(evaluationTest.toClassDetailsString());

	}
}