/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.neuroph.samples;

import java.util.Arrays;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

/**
 * This sample shows how to train MultiLayerPerceptron neural network for iris classification problem using Neuroph
 * For more details about training process, error, iterations use NeurophStudio which provides rich environment  for
 * training and inspecting neural networks
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class IrisClassificationSample {


    /**
     *  Runs this sample
     */
    public static void main(String[] args) {    
        // get the path to file with data
        String inputFileName = "data_sets/iris_data_normalised.txt";
        
        // create MultiLayerPerceptron neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(4, 16, 3);
        // create training set from file
        DataSet irisDataSet = DataSet.createFromFile(inputFileName, 4, 3, ",");

        // attach learningn listener to print out info about error at each iteration
        neuralNet.getLearningRule().addListener((event)->{
            BackPropagation bp = (BackPropagation) event.getSource();
            System.out.println("Current iteration: " + bp.getCurrentIteration());
            System.out.println("Error: " + bp.getTotalNetworkError());        
        });
        
        neuralNet.getLearningRule().setLearningRate(0.5);
        neuralNet.getLearningRule().setMaxError(0.01);
        neuralNet.getLearningRule().setMaxIterations(30000);

        // train the network with training set
        neuralNet.learn(irisDataSet);

        neuralNet.save("irisNet.nnet");
        
        System.out.println("Done training.");
        System.out.println("Testing network...");
    }
    

}