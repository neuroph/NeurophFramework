/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
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
package org.neuroph.contrib.rnn.example;

import org.neuroph.contrib.rnn.bptt.BackPropagationThroughTime;
import org.neuroph.contrib.rnn.util.SequenceModeller;
import org.neuroph.contrib.rnn.LSTM;
import org.neuroph.contrib.rnn.RNN;
import org.neuroph.contrib.rnn.bptt.LSTMBackPropagationThroughTime;
import org.neuroph.contrib.rnn.util.MatrixInitializer;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.eval.ErrorEvaluator;
import org.neuroph.eval.Evaluation;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class LSTMStockPricePredictionExample {

    public static void main(String[] args) {

        DataSet dataSet = DataSet.createFromFile("Google_Stock_Price.csv", 3, 1, ",");
        DataSet[] trainTestSplit = dataSet.split(0.8, 0.2);
        DataSet trainingSet = trainTestSplit[0];
        DataSet testSet = trainTestSplit[1];

        SequenceModeller sequenceModeller = new SequenceModeller(trainingSet);

        int inputsCount = sequenceModeller.getCharIndex().size();
        int hiddenCount = 100;

        int maxIterations = 100;
        double learningRate = 0.8;

        System.out.println("Creating neural network...");
        RNN lstm = new LSTM(inputsCount, hiddenCount, new MatrixInitializer(MatrixInitializer.Type.Uniform, 0.1, 0, 0));
        BackPropagationThroughTime bptt = new LSTMBackPropagationThroughTime();
        bptt.setLearningRate(learningRate);
        lstm.setLearningRule(bptt);

        System.out.println("Training network...");
        bptt.learn(trainingSet, maxIterations);
        System.out.println("Training completed.");

//        evaluate(lstm, testSet);
    }

    private static void evaluate(RNN lstm, DataSet testSet) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
