package org.neuroph.samples.crossvalidation;

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
import java.util.List;
import java.util.concurrent.ExecutionException;
import org.neuroph.eval.CrossFolds;
import org.neuroph.eval.CrossValidation;
import org.neuroph.eval.EvaluationResult;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;

/**
 *
 * @author Nevena Milenkovic
 */
/*
 INTRODUCTION TO THE PROBLEM AND DATA SET INFORMATION:
 1. Data set that will be used in this experiment: Wheat Seeds Dataset
    The Wheat Seeds Dataset involves the prediction of species given measurements of seeds from different varieties of wheat.
    The original data set that will be used in this experiment can be found at link: 
    http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt
2. Reference:  Magorzata Charytanowicz, Jerzy Niewczas ,Institute of Mathematics and Computer Science, ,The John Paul II Catholic University of Lublin, KonstantynÃ³w 1 H, ,PL 20-708 Lublin, Poland 
   Owner of database: Volker Lohweg (University of Applied Sciences, Ostwestfalen-Lippe, volker.lohweg '@' hs-owl.de) 
   M. Charytanowicz, J. Niewczas, P. Kulczycki, P.A. Kowalski, S. Lukasik, S. Zak, 'A Complete Gradient Clustering Algorithm for Features Analysis of X-ray Images', in: Information Technologies in Biomedicine, Ewa Pietka, Jacek Kawa (eds.), Springer-Verlag, Berlin-Heidelberg, 2010, pp. 15-24.
 
 
3. Number of instances: 210
4. Number of Attributes: 7 pluss class attributes
5. Attribute Information:    
 Inputs:
 7 attributes: 
 7 continuous feature values are computed for each seed:
 1) Area.
 2) Perimeter. 
 3) Compactness
 4) Length of kernel.
 5) Width of kernel.
 6) Asymmetry coefficient.
 7) Length of kernel groove.
 8) Output: Class variable (1, 2 or 3). Values indicate different varieties of wheat: Kama,Rosa and Canadian.
6. Missing Values: None.
 
 */
public class WheatSeeds implements LearningEventListener {

    public static void main(String[] args) throws InterruptedException, ExecutionException {
        (new WheatSeeds()).run();
    }

    public void run() throws InterruptedException, ExecutionException {
        System.out.println("Creating training set...");
        // get path to training set
        String trainingSetFileName = "data_sets/seeds.txt";
        int inputsCount = 7;
        int outputsCount = 3;

        // create training set from file
        DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, "\t");
        dataSet.shuffle();

        System.out.println("Creating neural network...");
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount, 15, 2, outputsCount);

        neuralNet.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule = (MomentumBackpropagation) neuralNet.getLearningRule();

        // set learning rate and max error
        learningRule.setLearningRate(0.1);
        learningRule.setMaxError(0.01);
        learningRule.setMaxIterations(1000);

        String[] classLabels = new String[]{"Cama", "Rosa", "Canadian"};
        neuralNet.setOutputLabels(classLabels);
        CrossValidation crossVal = new CrossValidation(neuralNet, dataSet, 10);
        EvaluationResult totalResult= crossVal.run();
         List<CrossFolds> cflist= crossVal.getFoldResults();
    }

    @Override
    public void handleLearningEvent(LearningEvent le) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
