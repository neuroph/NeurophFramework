package org.neuroph.samples.crossvalidation;

import java.util.List;
import java.util.concurrent.ExecutionException;
import org.neuroph.eval.KFoldCrossValidation;
import org.neuroph.eval.EvaluationResult;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.eval.FoldResult;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;

/**
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:

 1. Data set that will be used in this experiment: Iris Flower Dataset
The Iris Flowers Dataset involves predicting the flower species given measurements of iris flowers.
 The original data set that will be used in this experiment can be found at link:
http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

2. Reference:  R.A. Fisher
Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).

3. Number of instances: 150

4. Number of Attributes: 4 pluss class attributes

7. Attribute Information:
 Inputs:
 4 attributes:
 4 numerical features are computed for each flower:
1. Sepal length in cm.
2. Sepal width in cm.
3. Petal length in cm.
4. Petal width in cm.

Output:
Class (Iris Setosa, Iris Versicolour, Iris Virginica). They are represented as (1,0,0), (0,1,0) and (0,0,1) respectively.

8. Missing Values: None.




 */
public class IrisFlowers {

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        (new IrisFlowers()).run();
    }

    public void run() throws InterruptedException, ExecutionException {
        System.out.println("Creating training set...");
        // get path to training set
        String dataSetFile = "data_sets/iris_data_normalised.txt";
        int inputsCount = 4;
        int outputsCount = 3;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputsCount, outputsCount, ",");
     //   dataSet.setColumnNames(new String[]{"sepal.length", "sepal.width", "petal.length", "petal.width",  "setosa", "versicolor", "virginica"});
        dataSet.setColumnNames(new String[]{"setosa", "versicolor", "virginica"});
        dataSet.shuffle();

        System.out.println("Creating neural network...");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.TANH, inputsCount, 5, outputsCount);

        String[] classLabels = new String[]{"setosa", "versicolor", "virginica"};
        neuralNet.setOutputLabels(classLabels);

        KFoldCrossValidation crossVal = new KFoldCrossValidation(neuralNet, dataSet, 5);
        EvaluationResult totalResult= crossVal.run();
        List<FoldResult> cflist= crossVal.getResultsByFolds();

    }

}
