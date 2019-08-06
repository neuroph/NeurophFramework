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

import org.neuroph.contrib.rnn.util.LossFunction;
import org.neuroph.contrib.rnn.util.SequenceModeller;
import org.neuroph.contrib.rnn.lstm.LSTM;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.neuroph.core.data.DataSet;
import org.neuroph.nnet.learning.BackPropagation;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class BackPropagationThroughTime extends BackPropagation {

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

                LSTM lstm = (LSTM) this.getNeuralNetwork();
                Map<String, DoubleMatrix> valuesInTimesteps = new HashMap<>();

                for (int timestep = 0; timestep < seq.length() - 1; timestep++) {
                    DoubleMatrix input = charVector.get(String.valueOf(seq.charAt(timestep)));
                    valuesInTimesteps.put("input" + timestep, input);

                    lstm.activate(timestep, valuesInTimesteps);

                    DoubleMatrix predictedResult = lstm.decode(valuesInTimesteps.get("output" + timestep));
                    valuesInTimesteps.put("predictedResult" + timestep, predictedResult);
                    DoubleMatrix result = charVector.get(String.valueOf(seq.charAt(timestep + 1)));
                    valuesInTimesteps.put("result" + timestep, result);
                    error += LossFunction.getMeanCategoricalCrossEntropy(predictedResult, result);
                }

                BackPropagationThroughTime bptt = (BackPropagationThroughTime) lstm.getLearningRule();
                bptt.propagate(valuesInTimesteps, seq.length() - 2, bptt.getLearningRate());

                num += seq.length();
            }

            System.out.println("Iteration = " + (i + 1) + ", error = " + error / num + ", time = " + (System.currentTimeMillis() - start) / 1000 + "s");
        }
    }

    public void propagate(Map<String, DoubleMatrix> valuesInTimesteps, int lastTimestep, double learningRate) {
        LSTM lstm = (LSTM) this.getNeuralNetwork();

        for (int timestep = lastTimestep; timestep > -1; timestep--) {
            DoubleMatrix predictedResult = valuesInTimesteps.get("predictedResult" + timestep);
            DoubleMatrix result = valuesInTimesteps.get("result" + timestep);
            DoubleMatrix resultDelta = predictedResult.sub(result);
            valuesInTimesteps.put("resultDelta" + timestep, resultDelta);

            DoubleMatrix output = valuesInTimesteps.get("output" + timestep);
            
            DoubleMatrix outputDelta = null;
            if (timestep == lastTimestep) {
                outputDelta = computeOutputDeltaForLastTimestep(outputDelta, resultDelta, lstm);
            } else {
                outputDelta = computeOutputDeltaForNotLastTimestep(outputDelta, resultDelta, valuesInTimesteps, timestep, lstm);
            }

            valuesInTimesteps.put("outputDelta" + timestep, outputDelta);

            DoubleMatrix outputGate = valuesInTimesteps.get("outputActivationGate" + timestep);
            DoubleMatrix outputActivation = valuesInTimesteps.get("outputActivation" + timestep);
            DoubleMatrix outputActivationDelta = outputDelta.mul(outputGate).mul(deriveExp(outputActivation));
            valuesInTimesteps.put("outputActivationDelta" + timestep, outputActivationDelta);

            DoubleMatrix memoryCellDelta = null;
            if (timestep == lastTimestep) {
                memoryCellDelta = computeMemoryCellDeltaForLastTimestep(memoryCellDelta, outputDelta, outputActivation, outputActivationDelta, outputGate, lstm);
            } else {
                memoryCellDelta = computeMemoryCellDeltaForNotLastTimestep(memoryCellDelta, outputDelta, outputActivation, outputActivationDelta, outputGate, valuesInTimesteps, timestep, lstm);
            }
            valuesInTimesteps.put("memoryCellDelta" + timestep, memoryCellDelta);

            DoubleMatrix memoryCellGate = valuesInTimesteps.get("memoryCellGate" + timestep);
            DoubleMatrix inputActivation = valuesInTimesteps.get("inputActivation" + timestep);
            DoubleMatrix memoryCellGateDelta = memoryCellDelta.mul(inputActivation).mul(deriveTanh(memoryCellGate));
            valuesInTimesteps.put("memoryCellGateDelta" + timestep, memoryCellGateDelta);

            DoubleMatrix previousMemoryCellActivation = null;

            if (timestep > 0) {
                previousMemoryCellActivation = valuesInTimesteps.get("memoryCellActivation" + (timestep - 1));
            } else {
                previousMemoryCellActivation = DoubleMatrix.zeros(1, output.length);
            }

            DoubleMatrix forgetActivation = valuesInTimesteps.get("forgetActivation" + timestep);
            DoubleMatrix forgetActivationDelta = memoryCellDelta.mul(previousMemoryCellActivation).mul(deriveExp(forgetActivation));
            valuesInTimesteps.put("forgetActivationDelta" + timestep, forgetActivationDelta);

            DoubleMatrix inputActivationDelta = memoryCellDelta.mul(memoryCellGate).mul(deriveExp(inputActivation));
            valuesInTimesteps.put("inputActivationDelta" + timestep, inputActivationDelta);
        }

        updateParameters(valuesInTimesteps, lastTimestep, learningRate, lstm);
    }

    private void updateParameters(Map<String, DoubleMatrix> valuesInTimesteps, int lastTimestep, double learningRate, LSTM lstm) {
        DoubleMatrix inputGateInputWeightGate = new DoubleMatrix(lstm.getInputGateInputWeight().rows, lstm.getInputGateInputWeight().columns);
        DoubleMatrix inputGateOutputWeightGate = new DoubleMatrix(lstm.getInputGateOutputWeight().rows, lstm.getInputGateOutputWeight().columns);
        DoubleMatrix inputGateMemoryCellWeightGate = new DoubleMatrix(lstm.getInputGateMemoryCellWeight().rows, lstm.getInputGateMemoryCellWeight().columns);
        DoubleMatrix inputGateBiasGate = new DoubleMatrix(lstm.getInputGateBias().rows, lstm.getInputGateBias().columns);

        DoubleMatrix forgetGateInputWeightGate = new DoubleMatrix(lstm.getForgetGateInputWeight().rows, lstm.getForgetGateInputWeight().columns);
        DoubleMatrix forgetGateOutputWeightGate = new DoubleMatrix(lstm.getForgetGateOutputWeight().rows, lstm.getForgetGateOutputWeight().columns);
        DoubleMatrix forgetGateMemoryCellWeightGate = new DoubleMatrix(lstm.getForgetGateMemoryCellWeight().rows, lstm.getForgetGateMemoryCellWeight().columns);
        DoubleMatrix forgetGateBiasGate = new DoubleMatrix(lstm.getForgetGateBias().rows, lstm.getForgetGateBias().columns);

        DoubleMatrix memoryCellInputWeightGate = new DoubleMatrix(lstm.getMemoryCellInputWeight().rows, lstm.getMemoryCellInputWeight().columns);
        DoubleMatrix memoryCellOutputWeightGate = new DoubleMatrix(lstm.getMemoryCellOutputWeight().rows, lstm.getMemoryCellOutputWeight().columns);
        DoubleMatrix memoryCellBiasGate = new DoubleMatrix(lstm.getMemoryCellBias().rows, lstm.getMemoryCellBias().columns);

        DoubleMatrix outputGateInputWeightGate = new DoubleMatrix(lstm.getOutputGateInputWeight().rows, lstm.getOutputGateInputWeight().columns);
        DoubleMatrix outputGateOutputWeightGate = new DoubleMatrix(lstm.getOutputGateOutputWeight().rows, lstm.getOutputGateOutputWeight().columns);
        DoubleMatrix outputGateMemoryCellWeightGate = new DoubleMatrix(lstm.getOutputGateMemoryCellWeight().rows, lstm.getOutputGateMemoryCellWeight().columns);
        DoubleMatrix outputGateBiasGate = new DoubleMatrix(lstm.getOutputGateBias().rows, lstm.getOutputGateBias().columns);

        DoubleMatrix outputWeightGate = new DoubleMatrix(lstm.getOutputWeight().rows, lstm.getOutputWeight().columns);
        DoubleMatrix outputBiasGate = new DoubleMatrix(lstm.getOutputBias().rows, lstm.getOutputBias().columns);

        for (int timestep = 0; timestep < lastTimestep + 1; timestep++) {
            DoubleMatrix input = valuesInTimesteps.get("input" + timestep).transpose();
            inputGateInputWeightGate = inputGateInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("inputActivationDelta" + timestep)));
            forgetGateInputWeightGate = forgetGateInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("forgetActivationDelta" + timestep)));
            memoryCellInputWeightGate = memoryCellInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("memoryCellGateDelta" + timestep)));
            outputGateInputWeightGate = outputGateInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("outputActivationDelta" + timestep)));

            if (timestep > 0) {
                DoubleMatrix previousOutput = valuesInTimesteps.get("output" + (timestep - 1)).transpose();
                DoubleMatrix previousMemoryCellActivation = valuesInTimesteps.get("memoryCellActivation" + (timestep - 1)).transpose();
                inputGateOutputWeightGate = inputGateOutputWeightGate
                        .add(previousOutput.mmul(valuesInTimesteps.get("inputActivationDelta" + timestep)));
                forgetGateOutputWeightGate = forgetGateOutputWeightGate
                        .add(previousOutput.mmul(valuesInTimesteps.get("forgetActivationDelta" + timestep)));
                memoryCellOutputWeightGate = memoryCellOutputWeightGate
                        .add(previousOutput.mmul(valuesInTimesteps.get("memoryCellGateDelta" + timestep)));
                outputGateOutputWeightGate = outputGateOutputWeightGate
                        .add(previousOutput.mmul(valuesInTimesteps.get("outputActivationDelta" + timestep)));
                inputGateMemoryCellWeightGate = inputGateMemoryCellWeightGate
                        .add(previousMemoryCellActivation.mmul(valuesInTimesteps.get("inputActivationDelta" + timestep)));
                forgetGateMemoryCellWeightGate = forgetGateMemoryCellWeightGate
                        .add(previousMemoryCellActivation.mmul(valuesInTimesteps.get("forgetActivationDelta" + timestep)));
            }

            outputGateMemoryCellWeightGate = outputGateMemoryCellWeightGate
                    .add(valuesInTimesteps.get("memoryCellActivation" + timestep).transpose().mmul(valuesInTimesteps.get("outputActivationDelta" + timestep)));
            outputWeightGate = outputWeightGate
                    .add(valuesInTimesteps.get("output" + timestep).transpose().mmul(valuesInTimesteps.get("resultDelta" + timestep)));
            inputGateBiasGate = inputGateBiasGate
                    .add(valuesInTimesteps.get("inputActivationDelta" + timestep));
            forgetGateBiasGate = forgetGateBiasGate
                    .add(valuesInTimesteps.get("forgetActivationDelta" + timestep));
            memoryCellBiasGate = memoryCellBiasGate
                    .add(valuesInTimesteps.get("memoryCellGateDelta" + timestep));
            outputGateBiasGate = outputGateBiasGate
                    .add(valuesInTimesteps.get("outputActivationDelta" + timestep));
            outputBiasGate = outputBiasGate
                    .add(valuesInTimesteps.get("resultDelta" + timestep));
        }

        lstm.setInputGateInputWeight(lstm.getInputGateInputWeight().sub(inputGateInputWeightGate.div(lastTimestep).mul(learningRate)));
        lstm.setInputGateOutputWeight(lstm.getInputGateOutputWeight().sub(inputGateOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        lstm.setInputGateMemoryCellWeight(lstm.getInputGateMemoryCellWeight().sub(inputGateMemoryCellWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        lstm.setInputGateBias(lstm.getInputGateBias().sub(inputGateBiasGate.div(lastTimestep).mul(learningRate)));

        lstm.setForgetGateInputWeight(lstm.getForgetGateInputWeight().sub(forgetGateInputWeightGate.div(lastTimestep).mul(learningRate)));
        lstm.setForgetGateOutputWeight(lstm.getForgetGateOutputWeight().sub(forgetGateOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        lstm.setForgetGateMemoryCellWeight(lstm.getForgetGateMemoryCellWeight().sub(forgetGateMemoryCellWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        lstm.setForgetGateBias(lstm.getForgetGateBias().sub(forgetGateBiasGate.div(lastTimestep).mul(learningRate)));

        lstm.setMemoryCellInputWeight(lstm.getMemoryCellInputWeight().sub(memoryCellInputWeightGate.div(lastTimestep).mul(learningRate)));
        lstm.setMemoryCellOutputWeight(lstm.getMemoryCellOutputWeight().sub(memoryCellOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        lstm.setMemoryCellBias(lstm.getMemoryCellBias().sub(memoryCellBiasGate.div(lastTimestep).mul(learningRate)));

        lstm.setOutputGateInputWeight(lstm.getOutputGateInputWeight().sub(outputGateInputWeightGate.div(lastTimestep).mul(learningRate)));
        lstm.setOutputGateOutputWeight(lstm.getOutputGateOutputWeight().sub(outputGateOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        lstm.setOutputGateMemoryCellWeight(lstm.getOutputGateMemoryCellWeight().sub(outputGateMemoryCellWeightGate.div(lastTimestep).mul(learningRate)));
        lstm.setOutputGateBias(lstm.getOutputGateBias().sub(outputGateBiasGate.div(lastTimestep).mul(learningRate)));

        lstm.setOutputWeight(lstm.getOutputWeight().sub(outputWeightGate.div(lastTimestep).mul(learningRate)));
        lstm.setOutputBias(lstm.getOutputBias().sub(outputBiasGate.div(lastTimestep).mul(learningRate)));
    }

    private DoubleMatrix deriveExp(DoubleMatrix matrix) {
        return matrix.mul(DoubleMatrix.ones(1, matrix.length).sub(matrix));
    }

    private DoubleMatrix deriveTanh(DoubleMatrix matrix) {
        return DoubleMatrix.ones(1, matrix.length).sub(MatrixFunctions.pow(matrix, 2));
    }

    private DoubleMatrix computeOutputDeltaForLastTimestep(DoubleMatrix outputDelta, DoubleMatrix resultDelta, LSTM lstm) {
        outputDelta = lstm.getOutputWeight().mmul(resultDelta.transpose()).transpose();
        return outputDelta;
    }

    private DoubleMatrix computeOutputDeltaForNotLastTimestep(DoubleMatrix outputDelta, DoubleMatrix resultDelta, Map<String, DoubleMatrix> valuesInTimesteps, int timestep, LSTM lstm) {
        DoubleMatrix lateMemoryCellGateDelta = valuesInTimesteps.get("memoryCellGateDelta" + (timestep + 1));
        DoubleMatrix lateForgetActivationDelta = valuesInTimesteps.get("forgetActivationDelta" + (timestep + 1));
        DoubleMatrix lateOutputActivationDelta = valuesInTimesteps.get("outputActivationDelta" + (timestep + 1));
        DoubleMatrix lateInputActivationDelta = valuesInTimesteps.get("inputActivationDelta" + (timestep + 1));

        outputDelta = lstm.getOutputWeight().mmul(resultDelta.transpose()).transpose()
                .add(lstm.getMemoryCellOutputWeight().mmul(lateMemoryCellGateDelta.transpose()).transpose())
                .add(lstm.getInputGateOutputWeight().mmul(lateInputActivationDelta.transpose()).transpose())
                .add(lstm.getOutputGateOutputWeight().mmul(lateOutputActivationDelta.transpose()).transpose())
                .add(lstm.getForgetGateOutputWeight().mmul(lateForgetActivationDelta.transpose()).transpose());

        return outputDelta;
    }

    private DoubleMatrix computeMemoryCellDeltaForLastTimestep(DoubleMatrix memoryCellDelta, DoubleMatrix outputDelta, DoubleMatrix outputActivation, DoubleMatrix outputActivationDelta, DoubleMatrix outputGate, LSTM lstm) {
        memoryCellDelta = outputDelta.mul(outputActivation).mul(deriveTanh(outputGate))
                .add(lstm.getOutputGateMemoryCellWeight().mmul(outputActivationDelta.transpose()).transpose());
        return memoryCellDelta;
    }

    private DoubleMatrix computeMemoryCellDeltaForNotLastTimestep(DoubleMatrix memoryCellDelta, DoubleMatrix outputDelta, DoubleMatrix outputActivation, DoubleMatrix outputActivationDelta, DoubleMatrix outputGate, Map<String, DoubleMatrix> valuesInTimesteps, int timestep, LSTM lstm) {
        DoubleMatrix lateMemoryCellDelta = valuesInTimesteps.get("memoryCellDelta" + (timestep + 1));
        DoubleMatrix lateForgetActivationDelta = valuesInTimesteps.get("forgetActivationDelta" + (timestep + 1));
        DoubleMatrix lateForgetActivation = valuesInTimesteps.get("forgetActivation" + (timestep + 1));
        DoubleMatrix lateInputActivationDelta = valuesInTimesteps.get("inputActivationDelta" + (timestep + 1));

        memoryCellDelta = outputDelta.mul(outputActivation).mul(deriveTanh(outputGate))
                .add(lstm.getOutputGateMemoryCellWeight().mmul(outputActivationDelta.transpose()).transpose())
                .add(lateForgetActivation.mul(lateMemoryCellDelta))
                .add(lstm.getForgetGateMemoryCellWeight().mmul(lateForgetActivationDelta.transpose()).transpose())
                .add(lstm.getInputGateMemoryCellWeight().mmul(lateInputActivationDelta.transpose()).transpose());

        return memoryCellDelta;
    }

}
