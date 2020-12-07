package de.ukoeln.idh.teaching.jml.ex05;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.apache.commons.lang3.StringUtils;

import com.lexicalscope.jewel.cli.CliFactory;
import com.lexicalscope.jewel.cli.Option;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

public class Main {
	static String[][] parametersToTest = generateTestGrid();

	// this method generates potential parameter settings
	public static String[][] generateTestGrid() {
		List<String[]> r = new LinkedList<String[]>();

		// no pruning
		r.add(new String[] { "-U" });

		// default pruning
		double confidence = 0.05;
		do {
			for (int m = 2; m < 20; m++) {
				r.add(new String[] { "-C", String.valueOf(confidence), "-M", String.valueOf(m) });
				r.add(new String[] { "-C", String.valueOf(confidence), "-M", String.valueOf(m), "-B" });
			}
			confidence += 0.05;
		} while (confidence < 0.6);

		// reduced error pruning
		int folds = 2;
		do {
			for (int m = 2; m < 20; m++) {

				r.add(new String[] { "-R", "-N", String.valueOf(folds), "-M", String.valueOf(m) });
				r.add(new String[] { "-R", "-N", String.valueOf(folds), "-M", String.valueOf(m), "-B" });
			}

			folds++;
		} while (folds < 5);

		return r.toArray(new String[r.size()][]);
	};

	public static void main(String[] args) throws Exception {
		// parse command line options
		Options options = CliFactory.parseArguments(Options.class, args);

		// Initialize random number generator
		Random random = new Random(options.getRandomSeed());

		// Load training instances
		Instances instances = getData(options.getTrain());

		// split the training instances
		instances.randomize(random);
		int splitPoint = (int) Math.round(instances.numInstances() * 0.1);
		Instances dev = new Instances(instances, 0, splitPoint);
		Instances train = new Instances(instances, splitPoint, instances.numInstances() - splitPoint);

		// iterate over the parameter settings
		String[] maxParameters = null;
		double maxFscore = 0.0;
		for (String[] parameterSettings : parametersToTest) {
			System.err.print("Testing parameters: " + StringUtils.join(parameterSettings, " ") + ". ");
			double r = doRun(dev, copy(parameterSettings), random);
			System.err.println("Result: " + r);

			if (r >= maxFscore) {
				maxParameters = parameterSettings;
				maxFscore = r;
			}
		}

		// load test data
		Instances test = getData(options.getTest());

		// train the real classifier on train data
		J48 classifier = new J48();
		classifier.setOptions(copy(maxParameters));
		classifier.buildClassifier(train);

		// evaluate on test and print infos
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(classifier, test);
		System.err.println("Using parameters: " + StringUtils.join(maxParameters, " "));
		System.err.println(eval.toClassDetailsString());
		System.err.println(eval.toSummaryString());

	}

	// Create a copy of an array
	public static String[] copy(String[] arr) {
		return Arrays.copyOf(arr, arr.length);
	}

	// Loads a data set
	public static Instances getData(String fileName) throws IOException {
		File inputFile = new File(fileName);
		ArffLoader loader = new ArffLoader();
		loader.setFile(inputFile);
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		return instances;
	}

	// do one training with cross validation and return the mean f-score
	public static double doRun(Instances instances, String[] parameters, Random random) throws Exception {
		J48 classifier = new J48();
		classifier.setOptions(parameters);

		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 5, random);
		return eval.weightedFMeasure();
	}

	public interface Options {
		@Option(defaultValue = "1")
		Integer getRandomSeed();

		@Option
		String getTrain();

		@Option
		String getTest();

	}
}
