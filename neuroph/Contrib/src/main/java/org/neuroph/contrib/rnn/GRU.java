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
package org.neuroph.contrib.rnn;

import java.util.Map;
import org.jblas.DoubleMatrix;
import org.neuroph.contrib.rnn.util.Activation;
import org.neuroph.contrib.rnn.util.MatrixInitializer;
import org.neuroph.contrib.rnn.util.MatrixInitializer.Type;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public final class GRU extends RNN {

    private DoubleMatrix resetGateInputWeight;
    private DoubleMatrix resetGateOutputWeight;
    private DoubleMatrix resetGateBias;

    private DoubleMatrix updateGateInputWeight;
    private DoubleMatrix updateGateOutputWeight;
    private DoubleMatrix updateGateBias;

    private DoubleMatrix memoryCellInputWeight;
    private DoubleMatrix memoryCellOutputWeight;
    private DoubleMatrix memoryCellBias;

    private DoubleMatrix outputWeight;
    private DoubleMatrix outputBias;

    public GRU(int inputSize, int outputSize, MatrixInitializer matrixInitializer) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        if (matrixInitializer.getType() == Type.Uniform) {
            setUniformWeights(matrixInitializer);
        } else if (matrixInitializer.getType() == Type.Gaussian) {
            setGaussianWeights(matrixInitializer);
        }
    }

    public DoubleMatrix getResetGateInputWeight() {
        return resetGateInputWeight;
    }

    public void setResetGateInputWeight(DoubleMatrix resetGateInputWeight) {
        this.resetGateInputWeight = resetGateInputWeight;
    }

    public DoubleMatrix getResetGateOutputWeight() {
        return resetGateOutputWeight;
    }

    public void setResetGateOutputWeight(DoubleMatrix resetGateOutputWeight) {
        this.resetGateOutputWeight = resetGateOutputWeight;
    }

    public DoubleMatrix getResetGateBias() {
        return resetGateBias;
    }

    public void setResetGateBias(DoubleMatrix resetGateBias) {
        this.resetGateBias = resetGateBias;
    }

    public DoubleMatrix getUpdateGateInputWeight() {
        return updateGateInputWeight;
    }

    public void setUpdateGateInputWeight(DoubleMatrix updateGateInputWeight) {
        this.updateGateInputWeight = updateGateInputWeight;
    }

    public DoubleMatrix getUpdateGateOutputWeight() {
        return updateGateOutputWeight;
    }

    public void setUpdateGateOutputWeight(DoubleMatrix updateGateOutputWeight) {
        this.updateGateOutputWeight = updateGateOutputWeight;
    }

    public DoubleMatrix getUpdateGateBias() {
        return updateGateBias;
    }

    public void setUpdateGateBias(DoubleMatrix updateGateBias) {
        this.updateGateBias = updateGateBias;
    }

    public DoubleMatrix getMemoryCellInputWeight() {
        return memoryCellInputWeight;
    }

    public void setMemoryCellInputWeight(DoubleMatrix memoryCellInputWeight) {
        this.memoryCellInputWeight = memoryCellInputWeight;
    }

    public DoubleMatrix getMemoryCellOutputWeight() {
        return memoryCellOutputWeight;
    }

    public void setMemoryCellOutputWeight(DoubleMatrix memoryCellOutputWeight) {
        this.memoryCellOutputWeight = memoryCellOutputWeight;
    }

    public DoubleMatrix getMemoryCellBias() {
        return memoryCellBias;
    }

    public void setMemoryCellBias(DoubleMatrix memoryCellBias) {
        this.memoryCellBias = memoryCellBias;
    }

    public DoubleMatrix getOutputWeight() {
        return outputWeight;
    }

    public void setOutputWeight(DoubleMatrix outputWeight) {
        this.outputWeight = outputWeight;
    }

    public DoubleMatrix getOutputBias() {
        return outputBias;
    }

    public void setOutputBias(DoubleMatrix outputBias) {
        this.outputBias = outputBias;
    }

    @Override
    public void activate(int timestep, Map<String, DoubleMatrix> valuesInTimesteps) {
        DoubleMatrix input = valuesInTimesteps.get("input" + timestep);
        DoubleMatrix previousOutput = null;

        if (timestep == 0) {
            previousOutput = new DoubleMatrix(1, outputSize);
        } else {
            previousOutput = valuesInTimesteps.get("output" + (timestep - 1));
        }

        DoubleMatrix resetActivation = Activation.logistic(input.mmul(resetGateInputWeight)
                .add(previousOutput.mmul(resetGateOutputWeight))
                .add(resetGateBias));
        DoubleMatrix updateActivation = Activation.logistic(input.mmul(updateGateInputWeight)
                .add(previousOutput.mmul(updateGateOutputWeight))
                .add(updateGateBias));
        DoubleMatrix memoryCellGate = Activation.tanh(input.mmul(memoryCellInputWeight)
                .add(resetActivation.mul(previousOutput).mmul(memoryCellOutputWeight))
                .add(memoryCellBias));
        DoubleMatrix output = (DoubleMatrix.ones(1, updateActivation.columns)
                .sub(updateActivation)).mul(previousOutput)
                .add(updateActivation.mul(memoryCellGate));

        valuesInTimesteps.put("resetActivation" + timestep, resetActivation);
        valuesInTimesteps.put("updateActivation" + timestep, updateActivation);
        valuesInTimesteps.put("memoryCellGate" + timestep, memoryCellGate);
        valuesInTimesteps.put("output" + timestep, output);
    }

    @Override
    public DoubleMatrix decode(DoubleMatrix matrix) {
        return Activation.softmax(matrix.mmul(outputWeight).add(outputBias));
    }

    @Override
    protected void setUniformWeights(MatrixInitializer matrixInitializer) {
        this.resetGateInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.resetGateOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.resetGateBias = new DoubleMatrix(1, outputSize);

        this.updateGateInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.updateGateOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.updateGateBias = new DoubleMatrix(1, outputSize);

        this.memoryCellInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.memoryCellOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.memoryCellBias = new DoubleMatrix(1, outputSize);

        this.outputWeight = matrixInitializer.uniform(outputSize, inputSize);
        this.outputBias = new DoubleMatrix(1, inputSize);
    }

    @Override
    protected void setGaussianWeights(MatrixInitializer matrixInitializer) {
        this.resetGateInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.resetGateOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.resetGateBias = new DoubleMatrix(1, outputSize);

        this.updateGateInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.updateGateOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.updateGateBias = new DoubleMatrix(1, outputSize);

        this.memoryCellInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.memoryCellOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.memoryCellBias = new DoubleMatrix(1, outputSize);

        this.outputWeight = matrixInitializer.gaussian(outputSize, inputSize);
        this.outputBias = new DoubleMatrix(1, inputSize);
    }

}
