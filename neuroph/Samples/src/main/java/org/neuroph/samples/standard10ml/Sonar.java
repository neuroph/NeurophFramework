/**
 * Copyright 2013 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.neuroph.samples.standard10ml;

import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;

/**
 * Example of binary classification (mine or rock) using Multi Layer Perceptron on sonar signal data set.
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:

 1. Data set that will be used in this experiment: Sonar Dataset
    The Sonar Dataset involves the prediction of whether or not an object is a mine or a rock given the strength of sonar returns at different angles.
    The original data set that will be used in this experiment can be found at link:
    https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data

2. Reference:  Terry Sejnowski
   Gorman, R. P., and Sejnowski, T. J. (1988). "Analysis of Hidden Units in a Layered Network Trained to Classify Sonar Targets" in Neural Networks, Vol. 1, pp. 75-89.

3. Number of instances: 208

4. Number of Attributes: 60 pluss class attributes

5. Attribute Information:
   Inputs:
   60 attributes:
   Each input belongs to a set of 60 numbers in the range 0.0 to 1.0.
   Each number represents the energy within a particular frequency band, integrated over a certain period of time.
   1. - 60. Sonar returns at different angles

   Output: Class variable (0 or 1). Value 0 indicates that an object is a rock(R), and 1 that is a metal cylinder.

8. Missing Values: None.

 */
public class Sonar {

    public static void main(String[] args) {
        (new Sonar()).run();
    }

    public void run() {
        String dataSetFile = "data_sets/ml10standard/sonardata.txt";
        int numInputs = 60;
        int numOutputs = 1;

        // create data set from csv file
        DataSet dataSet = DataSet.createFromFile(dataSetFile, numInputs, numOutputs, ",");

        // split data into train and test set
        DataSet[] trainTestSplit = dataSet.split(0.8, 0.2);
        DataSet trainingSet = trainTestSplit[0];
        DataSet testSet = trainTestSplit[1];

        // create neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.TANH, numInputs, 10, numOutputs);

        // set learning rule and add listener
        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.addListener((event) -> {
            MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
            System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
        });

        // set learning rate and max error
        learningRule.setLearningRate(0.01);
        learningRule.setMaxError(0.01);
        
        learningRule.setMaxIterations(10000);
        learningRule.setMomentum(0.5);

        // train the network with training set
        neuralNet.learn(trainingSet);

        // evaluate network performance on test set
        evaluate(neuralNet, testSet);

        // save neural network to file
        neuralNet.save("nn1.nnet");

        System.out.println("Done.");

    }

    // Evaluates performance of neural network.
    // Contains calculation of Confusion matrix for classification tasks or Mean Ssquared Error and Mean Absolute Error for regression tasks.
    // Difference in binary and multi class classification are made when adding Evaluator (MultiClass or Binary).
    public void evaluate(NeuralNetwork neuralNet, DataSet dataSet) {

        System.out.println("Calculating performance indicators for neural network.");

        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));

        evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        evaluation.evaluate(neuralNet, dataSet);

        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.Binary.class);
        ConfusionMatrix confusionMatrix = evaluator.getResult();
        System.out.println("Confusion matrrix:\r\n");
        System.out.println(confusionMatrix.toString() + "\r\n\r\n");
        System.out.println("Classification metrics\r\n");
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats average = ClassificationMetrics.average(metrics);
        for (ClassificationMetrics cm : metrics) {
            System.out.println(cm.toString() + "\r\n");
        }
        System.out.println(average.toString());
    }

}
