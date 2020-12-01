package de.ukoeln.idh.teaching.jml.ex05;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

public class Main {

	public static void main(String[] args) throws Exception {
	
		 Instances importedData = new Instances(new BufferedReader(new FileReader("src/main/resources/germancredit/train.arff")));
		// System.out.println(importedData);
		 int numberOfFeatures = importedData.numAttributes();
		// System.out.println(numberOfFeatures);
		 
		 if (importedData.classIndex() == -1)
			 importedData.setClassIndex(importedData.numAttributes() - 1);
		
		 RemovePercentage splitTrain = new RemovePercentage();
		 splitTrain.setInvertSelection(true);
		 splitTrain.setPercentage(90.0);
		 splitTrain.setInputFormat(importedData); 
		 
		 Instances trainingData = new Instances(Filter.useFilter(importedData, splitTrain));
		 
		// System.out.println(trainingData);
		 
		 RemovePercentage splitPara = new RemovePercentage();
		 splitPara.setInvertSelection(true);
		 splitPara.setPercentage(90.0);
		 splitPara.setInputFormat(importedData); 
		 splitPara.setInvertSelection(false);
		 
		 
		 Instances paraTuningData = new Instances(Filter.useFilter(importedData, splitPara));
		 
		// System.out.println(paraTuningData);
		 
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingData);
		saver.setFile(new File("src/main/resources/germancredit/traindump.arff"));
		saver.writeBatch();
		
		ArffSaver saver2 = new ArffSaver();
		saver2.setInstances(paraTuningData);
		saver2.setFile(new File("src/main/resources/germancredit/paradump.arff"));
		saver2.writeBatch();
		
		
		RandomForest rf = new RandomForest();
		rf.setNumIterations(10);
		rf.setSeed(1);
		rf.setNumFeatures(newNumFeatures);
		
		
		
		
		 
	}
	
	public Instances splitter(Instances data, double percent) {
		
		
		
		return data;
	}

	
}