package de.ukoeln.idh.teaching.jml.ex05;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToNominal;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Date;
import java.util.Random;
public class CredibilityAnalyzer {
    Date date = new Date();

    /**
     *
     * @param path the path to the File which contains arff data
     * @return a Data Instances instance based on the arff file
     * @throws IOException
     */
    public Instances load(String path) throws IOException {
        FileReader file = new FileReader(path);
        BufferedReader data = new BufferedReader(file);
        return new Instances(data);
    }

    public Instances[] splitData(Instances data) {
        RemovePercentage filter = new RemovePercentage();
        filter.setPercentage(90);
        try {
            filter.setInputFormat(data);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        try {
            Instances[] instances = new Instances[2];
            Instances left = new Instances(Filter.useFilter(data, filter));
            filter.setInvertSelection(true);
            Instances right = new Instances(Filter.useFilter(data, filter));
            instances[0] = left;
            instances[1] = right;
            return instances;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     *
     * @param data
     * @param features An array of features that are supposed to be removed
     * @return a Data Instances instance based the data where features are removed
     */
    public Instances remove(Instances data, int[] features) {
        Remove remove = new Remove();
        String attributes = String.join(
                ",",
                Arrays.stream(features)
                        .mapToObj(String::valueOf) // map values to string object
                        .toArray(String[]::new) // create an string[] from map ;
        );
        remove.setAttributeIndices(attributes);
        remove.setInvertSelection(true); // revert
        try {
            remove.setInputFormat(data);
            return new Instances(Filter.useFilter(data, remove));
        } catch (Exception e) {
            System.out.println("Could not remove features: " + features);
            return data;
        }
    }

    /**
     *
     * @param data
     * @return a Data Instances instance based the data where string features are now nominal
     */
    public Instances toNominal(Instances data) {
        StringToNominal stringConverter = new StringToNominal();
        try {
            stringConverter.setInputFormat(data);
            String[] options = {"-R", "first-last"};
            stringConverter.setOptions(options);
            return new Instances(Filter.useFilter(data, stringConverter));
        } catch (Exception e) {
            System.out.println("Could not make features nominal" );
            return data;
        }
    }

    public Instances clearData(Instances data) {
        RemoveDuplicates removeDuplicates = new RemoveDuplicates();
        try {
            removeDuplicates.setInputFormat(data);
            return new Instances(Filter.useFilter(data, removeDuplicates));
        } catch (Exception e) {
            System.out.println("Could not clea data from duplicates" );
            return data;
        }
    }

    /**
     *
     * @param data the data instance which is supposed to be evaluated
     * @param targetIndex the target feature of the dataset
     * @param classifier the classifier which is supposed to be taken
     */
    public void evaluate(Instances data, int targetIndex, Classifier classifier)  {
        try {
            data.setClassIndex(targetIndex);
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier, data, 10, new Random(1));

            System.out.println("Estimated Accuracy: " + evaluation.toSummaryString());
            System.out.println(evaluation.toClassDetailsString());
            System.out.println(evaluation.toMatrixString());
        } catch (Exception e) {
            System.err.println("Could not evaluate with classifier " + classifier.toString() + " and targetIndex" + targetIndex);
            e.printStackTrace();
        }
    }

    double getF(Instances data, Classifier classifier) {
        try {
            Evaluation evaluation = new Evaluation(data);
            evaluation.crossValidateModel(classifier,
                    data,
                    10,
                    new Random(this.date.getTime())
            );
            double f = evaluation.weightedFMeasure();
            return f > 0 ? f : 0;
        } catch (Exception e) {
            System.err.println("Could not evaluate with classifier " + classifier.toString());
            return 0;
        }
    }
}
