package de.ukoeln.idh.teaching.jml.ex05;

public class ModelScore {
	double score;
	int numTrees, treeDepth, numFeatures; 
	String batchSize;
	boolean breakTiesRandomly;
	
	
	public ModelScore(double score_, int numTrees_, String batchSize_, int treeDepth_, int numFeatures_, boolean breakTiesRandomly_) {
		score = score_;
		numTrees = numTrees_;
		batchSize = batchSize_;
		treeDepth = treeDepth_;
		numFeatures = numFeatures_;
		breakTiesRandomly = breakTiesRandomly_;	
	}
	
	public void printModel() {
		System.out.println("---Model---");
		System.out.println("numTrees: " + numTrees);
		System.out.println("batchSize: " + batchSize);
		System.out.println("treeDepth: " + treeDepth);
		System.out.println("numFeatures: " + numFeatures);
		System.out.println("breakTiesRandomly: " + breakTiesRandomly);
		System.out.println("------");
		System.out.println("weighted F1 score: " + score);
	}
	
}