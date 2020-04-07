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
package org.neuroph.util.data.norm;

import java.io.Serializable;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

/**
 * MaxMin normalization method, which normalize data in regard to min and max
 * elements in training set (by columns) Normalization is done according to
 * formula: normalizedVector[i] = (vector[i] - min[i]) / (max[i] - min[i])
 *
 * This class works fine if max and min are both positive and we want to
 * normalize to [0,1]
 *
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public final class MaxMinNormalizer implements Normalizer, Serializable {

    private double[] maxIn, maxOut; // contains max values for in and out columns
    private double[] minIn, minOut; // contains min values for in and out columns
    
    /**
     * Creates a new MaxMinNormalizer which is initialized with min and max vales from the dataset specified as input arument.
     * Normalizes all input and output values.
     * @param dataSet data set to initialize min and max values 
     */
    public MaxMinNormalizer(DataSet dataSet) {
        init(dataSet);
    }

    @Override
    public void normalize(DataSet dataSet) {
        for (DataSetRow row : dataSet.getRows()) {
            normalizeVector(row.getInput(), minIn, maxIn);

            if (dataSet.isSupervised()) {
               normalizeVector(row.getDesiredOutput(), minOut, maxOut);
            }
        }
    }

   /**
    * Initialize normalizer: finds min and max values for all the columns in the data set.
    * 
    * @param dataSet 
    */
    private void init(DataSet dataSet) {
        int numInputs = dataSet.getInputSize();
        int numOutputs = dataSet.getOutputSize();

        maxIn = new double[numInputs];
        minIn = new double[numInputs];

        for (int i = 0; i < numInputs; i++) {
            maxIn[i] = Double.MIN_VALUE;
            minIn[i] = Double.MAX_VALUE;
        }

        maxOut = new double[numOutputs];
        minOut = new double[numOutputs];

        for (int i = 0; i < numOutputs; i++) {
            maxOut[i] = Double.MIN_VALUE;
            minOut[i] = Double.MAX_VALUE;
        }

        for (DataSetRow dataSetRow : dataSet.getRows()) {
            double[] input = dataSetRow.getInput();
            for (int i = 0; i < numInputs; i++) {
                if (Math.abs(input[i]) > maxIn[i]) {
                    maxIn[i] = Math.abs(input[i]);
                }
                if (Math.abs(input[i]) < minIn[i]) {
                    minIn[i] = Math.abs(input[i]);
                }
            }

            double[] output = dataSetRow.getDesiredOutput();
            for (int i = 0; i < numOutputs; i++) {
                if (Math.abs(output[i]) > maxOut[i]) {
                    maxOut[i] = Math.abs(output[i]);
                }
                if (Math.abs(output[i]) < minOut[i]) {
                    minOut[i] = Math.abs(output[i]);
                }
            }
        }
    }

    /**
     * Performs normalization of the given input vector.
     * 
     * @param vector vector to normalize
     * @param min vector of min values
     * @param max vector of max values
     */
    private void normalizeVector(double[] vector, double[] min, double[] max) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = (vector[i] - min[i]) / (max[i] - min[i]);
        }
    }

    public double[] getMaxIn() {
        return maxIn;
    }

    public double[] getMaxOut() {
        return maxOut;
    }

    public double[] getMinIn() {
        return minIn;
    }

    public double[] getMinOut() {
        return minOut;
    }

}
