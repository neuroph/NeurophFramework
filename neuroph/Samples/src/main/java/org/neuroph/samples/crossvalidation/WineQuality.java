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
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:
 1. Data set that will be used in this experiment: Wine Quality Dataset
    The Wine Quality Dataset involves predicting the quality of white wines on a scale given chemical measures of each wine.
    It is a multi-class classification problem, but could also be framed as a regression problem.
    The original data set that will be used in this experiment can be found at link:
    http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv
2. Reference:  National Institute of Diabetes and Digestive and Kidney Diseases
   Paulo Cortez, University of Minho, Guimarães, Portugal, http://www3.dsi.uminho.pt/pcortez
   A. CeA. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal , @ 2009

3. Number of instances: 4 898
4. Number of Attributes: 11 pluss class attributes (inputs are continuous aand numerical values, and output is numerical)
5. Attribute Information:
 Inputs:
 11 attributes:
 11 numerical or continuous features are computed for each wine:
 1) Fixed acidity.
 2) Volatile acidity.
 3) Citric acid.
 4) Residual sugar.
 5) Chlorides.
 6) Free sulfur dioxide.
 7) Total sulfur dioxide.
 8) Density.
 9) pH.
 10) Sulphates.
 11) Alcohol.
 12) Output: Quality (score between 0 and 10).
6. Missing Values: None.

 */
public class WineQuality implements LearningEventListener {

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        (new WineQuality()).run();
    }

    public void run() throws InterruptedException, ExecutionException {
        System.out.println("Creating training set...");
        // get path to training set
        String dataSetFile = "data_sets/wine.txt";
        int inputsCount = 11;
        int outputsCount = 10;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(dataSetFile, inputsCount, outputsCount, "\t", true);
        Normalizer norm = new MaxNormalizer(dataSet);
        norm.normalize(dataSet);
        dataSet.shuffle();

        System.out.println("Creating neural network...");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 20, 15, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max error
        learningRule.setLearningRate(0.1);
        learningRule.setMaxIterations(10);

        String classLabels[] = new String[]{"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"};
        neuralNet.setOutputLabels(classLabels);
        KFoldCrossValidation crossVal = new KFoldCrossValidation(neuralNet, dataSet, 10);
        EvaluationResult totalResult= crossVal.run();
        List<FoldResult> cflist= crossVal.getResultsByFolds();
    }

    @Override
    public void handleLearningEvent(LearningEvent event) {
    }
}
