# Exercise 5: Parameter Tuning


The goal of the second exercise is to implement parameter tuning with a held out development set.

## Step 1
Please `clone` the repository `https://github.com/idh-cologne-machine-learning-mit-java/exercise-05`.

Create a new branch, using your UzK username.

## Step 2
The `pom.xml` file already contains Weka as a dependency, such that you can directly start coding. 

As we have seen by now, even relatively simple algorithms like decision trees offer tuning parameters (so-called *hyperparameters*, because they are different than the feature weights learned within an algorithm). While there might be application-oriented arguments for some parameter settings (e.g., if you aim to apply your tree on specialised hardware, you might need to restrict the learning to binary branching), in general, finding the optimal parameter settings is an empirical question. I.e., one that needs to be solved on data.

However, we cannot really use the test data for this, because that would pollute them. We also cannot really use the training data for this, because that would lead to serious overfitting. We therefore need a third kind of data, often called *development data* or *tuning data*.

Using the dataset provided in `src/main/resources`, implement code to 

1. split the training data set in 90/10 sub sets, using the smaller sub set for parameter tuning, and the larger one for proper training.
2. systematically test "all" possible hyper parameter settings, using cross validation on the development data set. Find out which hyper parameters give you the best weighted average f-score (in the string output that Weka generates, this is in the line below the class-wise evaluation table -- but of course you should not parse the string output ...).
3. train a model on the remaining training data set, using the best performing hyper parameters.
4. Apply and evaluate it on the proper test data set.

Please use the `weka.classifiers.trees.RandomForest` algorithm for these experiments.

## Step 3
Commit your changes and push them to the server.