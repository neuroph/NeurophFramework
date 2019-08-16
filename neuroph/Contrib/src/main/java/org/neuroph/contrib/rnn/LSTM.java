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
public final class LSTM extends RNN {

    private DoubleMatrix inputGateInputWeight;
    private DoubleMatrix inputGateOutputWeight;
    private DoubleMatrix inputGateMemoryCellWeight;
    private DoubleMatrix inputGateBias;

    private DoubleMatrix forgetGateInputWeight;
    private DoubleMatrix forgetGateOutputWeight;
    private DoubleMatrix forgetGateMemoryCellWeight;
    private DoubleMatrix forgetGateBias;

    private DoubleMatrix memoryCellInputWeight;
    private DoubleMatrix memoryCellOutputWeight;
    private DoubleMatrix memoryCellBias;

    private DoubleMatrix outputGateInputWeight;
    private DoubleMatrix outputGateOutputWeight;
    private DoubleMatrix outputGateMemoryCellWeight;
    private DoubleMatrix outputGateBias;

    private DoubleMatrix outputWeight;
    private DoubleMatrix outputBias;

    public LSTM(int inputSize, int outputSize, MatrixInitializer matrixInitializer) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        if (matrixInitializer.getType() == Type.Uniform) {
            setUniformWeights(matrixInitializer);
        } else if (matrixInitializer.getType() == Type.Gaussian) {
            setGaussianWeights(matrixInitializer);
        }
    }

    public DoubleMatrix getInputGateInputWeight() {
        return inputGateInputWeight;
    }

    public void setInputGateInputWeight(DoubleMatrix inputGateInputWeight) {
        this.inputGateInputWeight = inputGateInputWeight;
    }

    public DoubleMatrix getInputGateOutputWeight() {
        return inputGateOutputWeight;
    }

    public void setInputGateOutputWeight(DoubleMatrix inputGateOutputWeight) {
        this.inputGateOutputWeight = inputGateOutputWeight;
    }

    public DoubleMatrix getInputGateMemoryCellWeight() {
        return inputGateMemoryCellWeight;
    }

    public void setInputGateMemoryCellWeight(DoubleMatrix inputGateMemoryCellWeight) {
        this.inputGateMemoryCellWeight = inputGateMemoryCellWeight;
    }

    public DoubleMatrix getInputGateBias() {
        return inputGateBias;
    }

    public void setInputGateBias(DoubleMatrix inputGateBias) {
        this.inputGateBias = inputGateBias;
    }

    public DoubleMatrix getForgetGateInputWeight() {
        return forgetGateInputWeight;
    }

    public void setForgetGateInputWeight(DoubleMatrix forgetGateInputWeight) {
        this.forgetGateInputWeight = forgetGateInputWeight;
    }

    public DoubleMatrix getForgetGateOutputWeight() {
        return forgetGateOutputWeight;
    }

    public void setForgetGateOutputWeight(DoubleMatrix forgetGateOutputWeight) {
        this.forgetGateOutputWeight = forgetGateOutputWeight;
    }

    public DoubleMatrix getForgetGateMemoryCellWeight() {
        return forgetGateMemoryCellWeight;
    }

    public void setForgetGateMemoryCellWeight(DoubleMatrix forgetGateMemoryCellWeight) {
        this.forgetGateMemoryCellWeight = forgetGateMemoryCellWeight;
    }

    public DoubleMatrix getForgetGateBias() {
        return forgetGateBias;
    }

    public void setForgetGateBias(DoubleMatrix forgetGateBias) {
        this.forgetGateBias = forgetGateBias;
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

    public DoubleMatrix getOutputGateInputWeight() {
        return outputGateInputWeight;
    }

    public void setOutputGateInputWeight(DoubleMatrix outputGateInputWeight) {
        this.outputGateInputWeight = outputGateInputWeight;
    }

    public DoubleMatrix getOutputGateOutputWeight() {
        return outputGateOutputWeight;
    }

    public void setOutputGateOutputWeight(DoubleMatrix outputGateOutputWeight) {
        this.outputGateOutputWeight = outputGateOutputWeight;
    }

    public DoubleMatrix getOutputGateMemoryCellWeight() {
        return outputGateMemoryCellWeight;
    }

    public void setOutputGateMemoryCellWeight(DoubleMatrix outputGateMemoryCellWeight) {
        this.outputGateMemoryCellWeight = outputGateMemoryCellWeight;
    }

    public DoubleMatrix getOutputGateBias() {
        return outputGateBias;
    }

    public void setOutputGateBias(DoubleMatrix outputGateBias) {
        this.outputGateBias = outputGateBias;
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
        DoubleMatrix previousOutputActivation = null;
        DoubleMatrix previousMemoryCellActivation = null;

        if (timestep == 0) {
            previousOutputActivation = new DoubleMatrix(1, outputSize);
            previousMemoryCellActivation = previousOutputActivation.dup();
        } else {
            previousOutputActivation = valuesInTimesteps.get("output" + (timestep - 1));
            previousMemoryCellActivation = valuesInTimesteps.get("memoryCellActivation" + (timestep - 1));
        }

        DoubleMatrix inputActivation = Activation.logistic(input.mmul(inputGateInputWeight)
                .add(previousOutputActivation.mmul(inputGateOutputWeight))
                .add(previousMemoryCellActivation.mmul(inputGateMemoryCellWeight))
                .add(inputGateBias));
        DoubleMatrix forgetActivation = Activation.logistic(input.mmul(forgetGateInputWeight)
                .add(previousOutputActivation.mmul(forgetGateOutputWeight))
                .add(previousMemoryCellActivation.mmul(forgetGateMemoryCellWeight))
                .add(forgetGateBias));
        DoubleMatrix memoryCellGate = Activation.tanh(input.mmul(memoryCellInputWeight)
                .add(previousOutputActivation.mmul(memoryCellOutputWeight))
                .add(memoryCellBias));
        DoubleMatrix memoryCellActivation = forgetActivation.mul(previousMemoryCellActivation)
                .add(inputActivation.mul(memoryCellGate));
        DoubleMatrix outputActivation = Activation.logistic(input.mmul(outputGateInputWeight)
                .add(previousOutputActivation.mmul(outputGateOutputWeight))
                .add(memoryCellActivation.mmul(outputGateMemoryCellWeight))
                .add(outputGateBias));
        DoubleMatrix outputActivationGate = Activation.tanh(memoryCellActivation);
        DoubleMatrix output = outputActivation.mul(outputActivationGate);

        valuesInTimesteps.put("inputActivation" + timestep, inputActivation);
        valuesInTimesteps.put("forgetActivation" + timestep, forgetActivation);
        valuesInTimesteps.put("memoryCellGate" + timestep, memoryCellGate);
        valuesInTimesteps.put("memoryCellActivation" + timestep, memoryCellActivation);
        valuesInTimesteps.put("outputActivation" + timestep, outputActivation);
        valuesInTimesteps.put("outputActivationGate" + timestep, outputActivationGate);
        valuesInTimesteps.put("output" + timestep, output);
    }

    @Override
    public DoubleMatrix decode(DoubleMatrix matrix) {
        return Activation.softmax(matrix.mmul(outputWeight).add(outputBias));
    }

    @Override
    protected void setUniformWeights(MatrixInitializer matrixInitializer) {
        this.inputGateInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.inputGateOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.inputGateMemoryCellWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.inputGateBias = new DoubleMatrix(1, outputSize);

        this.forgetGateInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.forgetGateOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.forgetGateMemoryCellWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.forgetGateBias = new DoubleMatrix(1, outputSize);

        this.memoryCellInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.memoryCellOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.memoryCellBias = new DoubleMatrix(1, outputSize);

        this.outputGateInputWeight = matrixInitializer.uniform(inputSize, outputSize);
        this.outputGateOutputWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.outputGateMemoryCellWeight = matrixInitializer.uniform(outputSize, outputSize);
        this.outputGateBias = new DoubleMatrix(1, outputSize);

        this.outputWeight = matrixInitializer.uniform(outputSize, inputSize);
        this.outputBias = new DoubleMatrix(1, inputSize);
    }

    @Override
    protected void setGaussianWeights(MatrixInitializer matrixInitializer) {
        this.inputGateInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.inputGateOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.inputGateMemoryCellWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.inputGateBias = new DoubleMatrix(1, outputSize);

        this.forgetGateInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.forgetGateOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.forgetGateMemoryCellWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.forgetGateBias = new DoubleMatrix(1, outputSize);

        this.memoryCellInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.memoryCellOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.memoryCellBias = new DoubleMatrix(1, outputSize);

        this.outputGateInputWeight = matrixInitializer.gaussian(inputSize, outputSize);
        this.outputGateOutputWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.outputGateMemoryCellWeight = matrixInitializer.gaussian(outputSize, outputSize);
        this.outputGateBias = new DoubleMatrix(1, outputSize);

        this.outputWeight = matrixInitializer.gaussian(outputSize, inputSize);
        this.outputBias = new DoubleMatrix(1, inputSize);
    }

}
