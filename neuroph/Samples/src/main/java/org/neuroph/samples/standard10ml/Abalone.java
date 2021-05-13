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
import java.util.List;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:

 1. Data set that will be used in this experiment: Abalone Dataset
    The Abalone Dataset involves predicting the age of abalone given objective measures of individuals.
    It is a multi-class classification problem, but can also be framed as a regression.
    The original data set that will be used in this experiment can be found at link:
    https://www.math.muni.cz/~kolacek/docs/frvs/M7222/data/AutoInsurSweden.txt

2. Reference: Marine Resources Division, Marine Research Laboratories - Taroona ,Department of Primary Industry and Fisheries, Tasmania ,GPO Box 619F, Hobart, Tasmania 7001, Australia
   Warwick J Nash, Tracy L Sellers, Simon R Talbot, Andrew J Cawthorn and Wes B Ford (1994)
   "The Population Biology of Abalone (_Haliotis_ species) in Tasmania. I. Blacklip Abalone (_H. rubra_) from the North Coast and Islands of Bass Strait",
   Sea Fisheries Division, Technical Report No. 48 (ISSN 1034-3288)

3. Number of instances: 4 177

4. Number of Attributes: 8 plus class attribute

5. Attribute Information:
 Inputs:
 8 attributes:
 8 features are computed for each abalone:
 1) Sex (M, F, I), which are represented as numerical values of 1,2,3 respectively.
 2) Length.
 3) Diameter.
 4) Height.
 5) Whole weight.
 6) Shucked weight.
 7) Viscera weight.
 8) Shell weight.

 9) Output: Rings mesaurment, numerical value.


6. Missing Values: none.




 */
public class Abalone {

    public static void main(String[] args) {
        (new Abalone()).run();
    }

    public void run() {
        System.out.println("Creating data set...");
        String dataSetFile = "data_sets/ml10standard/abalonerings.txt";
        int inputsCount = 8;
        int outputsCount = 29;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputsCount, outputsCount, "\t", true);
        DataSet[] trainTestSplit = dataSet.split(0.7, 0.3);
        DataSet trainingSet = trainTestSplit[0];
        DataSet testSet = trainTestSplit[1];

        Normalizer norm = new MaxNormalizer(trainingSet);
        norm.normalize(trainingSet);
        norm.normalize(testSet);

        System.out.println("Creating neural network...");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 15, 10, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();
        learningRule.addListener((event) -> {
            MomentumBackpropagation bp = (MomentumBackpropagation) event.getSource();
            System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());        
        });

        // set learning rate and max error
        learningRule.setLearningRate(0.1);
        learningRule.setMaxIterations(5000);

        System.out.println("Training network...");
        // train the network with training set
        neuralNet.learn(trainingSet);
        System.out.println("Training completed.");

        System.out.println("Saving network");
        // save neural network to file
        neuralNet.save("nn1.nnet");

        System.out.println("Done.");
    }





}
