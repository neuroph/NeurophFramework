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
package org.neuroph.contrib.rnn.bptt;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.neuroph.contrib.rnn.RNN;
import org.neuroph.contrib.rnn.util.LossFunction;
import org.neuroph.contrib.rnn.util.SequenceModeller;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.learning.BackPropagation;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public abstract class BackPropagationThroughTime extends BackPropagation {

    @Override
    public void learn(DataSet trainingSet, int maxIterations) {
        SequenceModeller sequenceModeller = new SequenceModeller(trainingSet);
        Map<String, DoubleMatrix> charVector = sequenceModeller.getCharVector();
        List<String> sequence = sequenceModeller.getSequence();

        for (int i = 0; i < maxIterations; i++) {
            double error = 0;
            double num = 0;
            double start = System.currentTimeMillis();

            for (int j = 0; j < sequence.size(); j++) {
                String seq = sequence.get(j);

                if (seq.length() < 3) {
                    continue;
                }

                RNN rnn = (RNN) this.getNeuralNetwork();
                Map<String, DoubleMatrix> valuesInTimesteps = new HashMap<>();

                for (int timestep = 0; timestep < seq.length() - 1; timestep++) {
                    DoubleMatrix input = charVector.get(String.valueOf(seq.charAt(timestep)));
                    valuesInTimesteps.put("input" + timestep, input);

                    rnn.activate(timestep, valuesInTimesteps);

                    DoubleMatrix predictedResult = rnn.decode(valuesInTimesteps.get("output" + timestep));
                    valuesInTimesteps.put("predictedResult" + timestep, predictedResult);
                    DoubleMatrix result = charVector.get(String.valueOf(seq.charAt(timestep + 1)));
                    valuesInTimesteps.put("result" + timestep, result);

                    error += LossFunction.getMeanCategoricalCrossEntropy(predictedResult, result);
                }

                BackPropagationThroughTime bptt = (BackPropagationThroughTime) rnn.getLearningRule();
                bptt.propagate(valuesInTimesteps, seq.length() - 2, bptt.getLearningRate());

                num += seq.length();
            }

            System.out.println("Iteration = " + (i + 1) + ", error = " + error / num + ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
        }
    }

    public abstract void propagate(Map<String, DoubleMatrix> valuesInTimesteps, int lastTimestep, double learningRate);

    protected abstract void updateParameters(Map<String, DoubleMatrix> valuesInTimesteps, int lastTimestep, double learningRate, RNN rnn);

    protected DoubleMatrix deriveExp(DoubleMatrix matrix) {
        return matrix.mul(DoubleMatrix.ones(1, matrix.length).sub(matrix));
    }

    protected DoubleMatrix deriveTanh(DoubleMatrix matrix) {
        return DoubleMatrix.ones(1, matrix.length).sub(MatrixFunctions.pow(matrix, 2));
    }

}
