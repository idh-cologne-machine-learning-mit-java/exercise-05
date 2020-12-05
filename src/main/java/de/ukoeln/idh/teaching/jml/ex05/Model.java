package de.ukoeln.idh.teaching.jml.ex05;

public class Model {
	int numIterations; 
	int batchSize;
	int maxDepth;
	int numFeatures;
	boolean attributeImportance; 
	boolean breakTies;
	double fmeasure;
	
	
	public Model(int numIterations, int batchSize, int maxDepth, int numFeatures, boolean attributeImportance, boolean breakTies, double fmeasure) {
		super();
		this.numIterations = numIterations;
		this.batchSize = batchSize;
		this.maxDepth = maxDepth;
		this.numFeatures = numFeatures;
		this.fmeasure = fmeasure;
		this.attributeImportance = attributeImportance;
		this.breakTies = breakTies;
		
	}
	
	public void printModel() {
		System.out.println("numIterations= " + numIterations);
		System.out.println("batchSize= " + batchSize);
		System.out.println("maxDepth= " + maxDepth);
		System.out.println("numFeatures= " + numFeatures);
		System.out.println("compute attribute importance= " +attributeImportance);
		System.out.println("randomly break ties= " + breakTies);
		System.out.println("fmeasure= " + fmeasure);
	}

}
