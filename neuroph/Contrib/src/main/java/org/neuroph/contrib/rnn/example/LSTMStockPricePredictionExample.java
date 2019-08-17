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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jblas.DoubleMatrix;
import org.neuroph.contrib.rnn.bptt.BackPropagationThroughTime;
import org.neuroph.contrib.rnn.util.SequenceModeller;
import org.neuroph.contrib.rnn.LSTM;
import org.neuroph.contrib.rnn.RNN;
import org.neuroph.contrib.rnn.bptt.LSTMBackPropagationThroughTime;
import org.neuroph.contrib.rnn.util.LossFunction;
import org.neuroph.contrib.rnn.util.MatrixInitializer;
import org.neuroph.core.data.DataSet;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class LSTMStockPricePredictionExample {

    public static void main(String[] args) {
        trainNetwork();
    }

    private static void trainNetwork() {
        DataSet trainingSet = DataSet.createFromFile("google-stock-price-train.csv", 3, 1, ",");
        DataSet testSet = DataSet.createFromFile("google-stock-price-test.csv", 3, 1, ",");

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

        testNetwork(lstm, testSet);
    }

    private static void testNetwork(RNN lstm, DataSet testSet) {
        SequenceModeller sequenceModeller = new SequenceModeller(testSet);
        Map<Integer, String> indexChar = sequenceModeller.getIndexChar();
        Map<String, DoubleMatrix> charVector = sequenceModeller.getCharVector();
        List<String> sequence = sequenceModeller.getSequence();

        System.out.println("Test set:");
        testSet.forEach(System.out::println);

        System.out.println("Prediction:");
        double error = 0;
        double num = 0;
        double start = System.currentTimeMillis();

        for (int j = 0; j < sequence.size(); j++) {
            String seq = sequence.get(j);

            Map<String, DoubleMatrix> valuesInTimesteps = new HashMap<>();

            System.out.print(String.valueOf(seq.charAt(0)));
            for (int timestep = 0; timestep < seq.length() - 1; timestep++) {
                DoubleMatrix input = charVector.get(String.valueOf(seq.charAt(timestep)));
                valuesInTimesteps.put("input" + timestep, input);

                lstm.activate(timestep, valuesInTimesteps);

                DoubleMatrix predictedResult = lstm.decode(valuesInTimesteps.get("output" + timestep));
                valuesInTimesteps.put("predictedResult" + timestep, predictedResult);
                DoubleMatrix result = charVector.get(String.valueOf(seq.charAt(timestep + 1)));
                valuesInTimesteps.put("result" + timestep, result);

                System.out.print(indexChar.get(predictedResult.argmax()));
                error += LossFunction.getMeanCategoricalCrossEntropy(predictedResult, result);
            }
            System.out.println();

            BackPropagationThroughTime bptt = (BackPropagationThroughTime) lstm.getLearningRule();
            bptt.propagate(valuesInTimesteps, seq.length() - 2, bptt.getLearningRate());

            num += seq.length();
        }

        System.out.println("Error = " + error / num + ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
    }

}
