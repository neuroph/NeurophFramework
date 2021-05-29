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

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
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
 * Example of simple multi class classification problem using iris flower data set.
 *
 * @author Nevena Milenkovic
 * @author Zoran Sevarac

 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:

 1. Data set that will be used in this experiment: Iris Flower Dataset
    The Iris Flowers Dataset involves predicting the flower species given measurements of iris flowers.
    The original data set that will be used in this experiment can be found at link:
    http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

2. Reference:  R.A. Fisher
   Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to Mathematical Statistics" (John Wiley, NY, 1950).

3. Number of instances: 150

4. Number of Attributes: 4 pluss class attributes

5. Attribute Information:
 Inputs:
 4 attributes:
 4 numerical features are computed for each flower:
 1) Sepal length in cm.
 2) Sepal width in cm.
 3) Petal length in cm.
 4) Petal width in cm.

 5)Output: Class (Iris Setosa, Iris Versicolour, Iris Virginica). They are represented as (1,0,0), (0,1,0) and (0,0,1) respectively.

6. Missing Values: None.
 */

public class IrisFlowers {

    public static void main(String[] args) {
        (new IrisFlowers()).run();
    }

    public void run() {
        System.out.println("Creating data set...");
        String dataSetFile = "data_sets/ml10standard/irisdatanormalised.txt";
        int inputsCount = 4;
        int outputsCount = 3;

        // create data set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputsCount, outputsCount, ",");

        // split data into training and test set
        DataSet[] trainTestSplit = dataSet.split(0.6, 0.4);
        DataSet trainingSet = trainTestSplit[0];
        DataSet testSet = trainTestSplit[1];

        System.out.println("Creating neural network...");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(TransferFunctionType.TANH, inputsCount, 16, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.addListener((event)->{
            if (event.getEventType() != LearningEvent.LEARNING_STOPPED) {
                MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
                System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
            }
        });

        // set learning rate and max error
        learningRule.setLearningRate(0.7);
        learningRule.setMomentum(0.9);
        learningRule.setMaxError(0.03);
    //    learningRule.setMaxIterations(10000);
        
        System.out.println("Training network...");
        // train the network with training set
        neuralNet.learn(trainingSet);
        System.out.println("Training completed.");
        System.out.println("Testing network...");

        System.out.println("Network performance on the test set");
        evaluate(neuralNet, testSet);

        System.out.println("Saving network");
        // save neural network to file
        neuralNet.save("nn1.nnet");

        System.out.println("Done.");
    }

    /**
     * Evaluates classification performance of a neural network.
     * Contains calculation of Confusion matrix for classification tasks or Mean Ssquared Error and Mean Absolute Error for regression tasks.
     *
     * @param neuralNet
     * @param testSet
     */
    public void evaluate(NeuralNetwork neuralNet, DataSet testSet) {

        System.out.println("Calculating performance indicators for neural network.");

        Evaluation evaluation = new Evaluation();
        evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));

        String[] classLabels = new String[]{"Setosa", "Virginica",  "Versicolor"};
        evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(classLabels));
        evaluation.evaluate(neuralNet, testSet);

        ClassifierEvaluator evaluator = evaluation.getEvaluator(ClassifierEvaluator.MultiClass.class);
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
