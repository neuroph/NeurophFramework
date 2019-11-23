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

import java.util.Map;
import org.jblas.DoubleMatrix;
import org.neuroph.contrib.rnn.GRU;
import org.neuroph.contrib.rnn.RNN;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class GRUBackPropagationThroughTime extends BackPropagationThroughTime {

    @Override
    public void propagate(Map<String, DoubleMatrix> valuesInTimesteps, int lastTimestep, double learningRate) {
        GRU gru = (GRU) this.getNeuralNetwork();

        for (int timestep = lastTimestep; timestep >= 0; timestep--) {
            DoubleMatrix predictedResult = valuesInTimesteps.get("predictedResult" + timestep);
            DoubleMatrix result = valuesInTimesteps.get("result" + timestep);
            DoubleMatrix resultDelta = predictedResult.sub(result);
            valuesInTimesteps.put("resultDelta" + timestep, resultDelta);

            DoubleMatrix output = valuesInTimesteps.get("output" + timestep);
            DoubleMatrix updateActivation = valuesInTimesteps.get("updateActivation" + timestep);
            DoubleMatrix resetActivation = valuesInTimesteps.get("resetActivation" + timestep);
            DoubleMatrix memoryCellGate = valuesInTimesteps.get("memoryCellGate" + timestep);

            DoubleMatrix outputDelta = null;
            if (timestep == lastTimestep) {
                outputDelta = computeOutputDeltaForLastTimestep(outputDelta, resultDelta, gru);
            } else {
                outputDelta = computeOutputDeltaForNotLastTimeStep(outputDelta, resultDelta, valuesInTimesteps, timestep, gru);

            }
            valuesInTimesteps.put("outputDelta" + timestep, outputDelta);

            DoubleMatrix memoryCellGateDelta = outputDelta.mul(updateActivation).mul(deriveTanh(memoryCellGate));
            valuesInTimesteps.put("memoryCellGateDelta" + timestep, memoryCellGateDelta);

            DoubleMatrix previousOutput = null;
            if (timestep > 0) {
                previousOutput = valuesInTimesteps.get("output" + (timestep - 1));
            } else {
                previousOutput = DoubleMatrix.zeros(1, output.length);
            }

            DoubleMatrix resetActivationDelta = (gru.getMemoryCellOutputWeight().mmul(memoryCellGateDelta.mul(previousOutput).transpose()).transpose()).mul(deriveExp(resetActivation));
            valuesInTimesteps.put("resetActivationDelta" + timestep, resetActivationDelta);

            DoubleMatrix updateActivationDelta = outputDelta.mul(memoryCellGate.sub(previousOutput)).mul(deriveExp(updateActivation));
            valuesInTimesteps.put("updateActivationDelta" + timestep, updateActivationDelta);
        }

        updateParameters(valuesInTimesteps, lastTimestep, learningRate, gru);
    }

    @Override
    protected void updateParameters(Map<String, DoubleMatrix> valuesInTimesteps, int lastTimestep, double learningRate, RNN rnn) {
        GRU gru = (GRU) rnn;

        DoubleMatrix resetGateInputWeightGate = new DoubleMatrix(gru.getResetGateInputWeight().rows, gru.getResetGateInputWeight().columns);
        DoubleMatrix resetGateOutputWeightGate = new DoubleMatrix(gru.getResetGateOutputWeight().rows, gru.getResetGateOutputWeight().columns);
        DoubleMatrix resetGateBiasGate = new DoubleMatrix(gru.getResetGateBias().rows, gru.getResetGateBias().columns);

        DoubleMatrix updateGateInputWeightGate = new DoubleMatrix(gru.getUpdateGateInputWeight().rows, gru.getUpdateGateInputWeight().columns);
        DoubleMatrix updateGateOutputWeightGate = new DoubleMatrix(gru.getUpdateGateOutputWeight().rows, gru.getUpdateGateOutputWeight().columns);
        DoubleMatrix updateGateBiasGate = new DoubleMatrix(gru.getUpdateGateBias().rows, gru.getUpdateGateBias().columns);

        DoubleMatrix memoryCellInputWeightGate = new DoubleMatrix(gru.getMemoryCellInputWeight().rows, gru.getMemoryCellInputWeight().columns);
        DoubleMatrix memoryCellOutputWeightGate = new DoubleMatrix(gru.getMemoryCellOutputWeight().rows, gru.getMemoryCellOutputWeight().columns);
        DoubleMatrix memoryCellBiasGate = new DoubleMatrix(gru.getMemoryCellBias().rows, gru.getMemoryCellBias().columns);

        DoubleMatrix outputWeightGate = new DoubleMatrix(gru.getOutputWeight().rows, gru.getOutputWeight().columns);
        DoubleMatrix outputBiasGate = new DoubleMatrix(gru.getOutputBias().rows, gru.getOutputBias().columns);

        for (int timestep = 0; timestep < lastTimestep + 1; timestep++) {
            DoubleMatrix input = valuesInTimesteps.get("input" + timestep).transpose();
            resetGateInputWeightGate = resetGateInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("resetActivationDelta" + timestep)));
            updateGateInputWeightGate = updateGateInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("updateActivationDelta" + timestep)));
            memoryCellInputWeightGate = memoryCellInputWeightGate
                    .add(input.mmul(valuesInTimesteps.get("memoryCellGateDelta" + timestep)));

            if (timestep > 0) {
                DoubleMatrix previousOutput = valuesInTimesteps.get("output" + (timestep - 1)).transpose();
                resetGateOutputWeightGate = resetGateOutputWeightGate
                        .add(previousOutput.mmul(valuesInTimesteps.get("resetActivationDelta" + timestep)));
                updateGateOutputWeightGate = updateGateOutputWeightGate
                        .add(previousOutput.mmul(valuesInTimesteps.get("updateActivationDelta" + timestep)));
                memoryCellOutputWeightGate = memoryCellOutputWeightGate
                        .add(valuesInTimesteps.get("resetActivation" + timestep).transpose().mul(previousOutput).mmul(valuesInTimesteps.get("memoryCellGateDelta" + timestep)));
            }
            outputWeightGate = outputWeightGate.add(valuesInTimesteps.get("output" + timestep).transpose().mmul(valuesInTimesteps.get("resultDelta" + timestep)));

            resetGateBiasGate = resetGateBiasGate.add(valuesInTimesteps.get("resetActivationDelta" + timestep));
            updateGateBiasGate = updateGateBiasGate.add(valuesInTimesteps.get("updateActivationDelta" + timestep));
            memoryCellBiasGate = memoryCellBiasGate.add(valuesInTimesteps.get("memoryCellGateDelta" + timestep));
            outputBiasGate = outputBiasGate.add(valuesInTimesteps.get("resultDelta" + timestep));
        }

        gru.setResetGateInputWeight(gru.getResetGateInputWeight().sub(resetGateInputWeightGate.div(lastTimestep).mul(learningRate)));
        gru.setResetGateOutputWeight(gru.getResetGateOutputWeight().sub(resetGateOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        gru.setResetGateBias(gru.getResetGateBias().sub(resetGateBiasGate.div(lastTimestep).mul(learningRate)));

        gru.setUpdateGateInputWeight(gru.getUpdateGateInputWeight().sub(updateGateInputWeightGate.div(lastTimestep).mul(learningRate)));
        gru.setUpdateGateOutputWeight(gru.getUpdateGateOutputWeight().sub(updateGateOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        gru.setUpdateGateBias(gru.getUpdateGateBias().sub(updateGateBiasGate.div(lastTimestep).mul(learningRate)));

        gru.setMemoryCellInputWeight(gru.getMemoryCellInputWeight().sub(memoryCellInputWeightGate.div(lastTimestep).mul(learningRate)));
        gru.setMemoryCellOutputWeight(gru.getMemoryCellOutputWeight().sub(memoryCellOutputWeightGate.div(lastTimestep < 2 ? 1 : (lastTimestep - 1)).mul(learningRate)));
        gru.setMemoryCellBias(gru.getMemoryCellBias().sub(memoryCellBiasGate.div(lastTimestep).mul(learningRate)));

        gru.setOutputWeight(gru.getOutputWeight().sub(outputWeightGate.div(lastTimestep).mul(learningRate)));
        gru.setOutputBias(gru.getOutputBias().sub(outputBiasGate.div(lastTimestep).mul(learningRate)));
    }

    private DoubleMatrix computeOutputDeltaForLastTimestep(DoubleMatrix outputDelta, DoubleMatrix resultDelta, GRU gru) {
        outputDelta = gru.getOutputWeight().mmul(resultDelta.transpose()).transpose();
        return outputDelta;
    }

    private DoubleMatrix computeOutputDeltaForNotLastTimeStep(DoubleMatrix outputDelta, DoubleMatrix resultDelta, Map<String, DoubleMatrix> valuesInTimesteps, int timestep, GRU gru) {
        DoubleMatrix lateOutputDelta = valuesInTimesteps.get("outputDelta" + (timestep + 1));
        DoubleMatrix lateMemoryCellGateDelta = valuesInTimesteps.get("memoryCellGateDelta" + (timestep + 1));
        DoubleMatrix lateResetActivationDelta = valuesInTimesteps.get("resetActivationDelta" + (timestep + 1));
        DoubleMatrix lateUpdateActivationDelta = valuesInTimesteps.get("updateActivationDelta" + (timestep + 1));
        DoubleMatrix lateResetActivation = valuesInTimesteps.get("resetActivation" + (timestep + 1));
        DoubleMatrix lateUpdateActivation = valuesInTimesteps.get("updateActivation" + (timestep + 1));

        outputDelta = gru.getOutputWeight().mmul(resultDelta.transpose()).transpose()
                .add(gru.getResetGateOutputWeight().mmul(lateResetActivationDelta.transpose()).transpose())
                .add(gru.getUpdateGateOutputWeight().mmul(lateUpdateActivationDelta.transpose()).transpose())
                .add(gru.getMemoryCellOutputWeight().mmul(lateMemoryCellGateDelta.mul(lateResetActivation).transpose()).transpose())
                .add(lateOutputDelta.mul(DoubleMatrix.ones(1, lateUpdateActivation.columns).sub(lateUpdateActivation)));

        return outputDelta;
    }

}
