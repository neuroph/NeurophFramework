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

import java.io.Serializable;
import java.util.Map;
import org.jblas.DoubleMatrix;
import org.neuroph.contrib.rnn.util.MatrixInitializer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.learning.BackPropagation;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public abstract class RNN extends NeuralNetwork<BackPropagation> implements Serializable {

    protected int inputSize;
    protected int outputSize;

    public int getInputSize() {
        return inputSize;
    }

    public void setInputSize(int inputSize) {
        this.inputSize = inputSize;
    }

    public int getOutputSize() {
        return outputSize;
    }

    public void setOutputSize(int outputSize) {
        this.outputSize = outputSize;
    }

    public abstract void activate(int timestep, Map<String, DoubleMatrix> valuesInTimesteps);

    public abstract DoubleMatrix decode(DoubleMatrix matrix);

    protected abstract void setUniformWeights(MatrixInitializer matrixInitializer);

    protected abstract void setGaussianWeights(MatrixInitializer matrixInitializer);

}
