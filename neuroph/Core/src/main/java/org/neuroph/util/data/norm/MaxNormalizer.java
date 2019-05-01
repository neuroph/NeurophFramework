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
 * Max normalization method, which normalize data in regard to max element in training set (by columns)
 * Normalization is done according to formula:
 * normalizedVector[i] = vector[i] / abs(max[i])
 *
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class MaxNormalizer implements Normalizer, Serializable {

    /**
     * Max values for input and output vectors
     */
    private double[] maxIn, maxOut;

    /**
     * Flag to indicate that normalizer is initialized
     */
    private boolean intialized=false;

    /**
     * Creates new instance of MaxNormalizer initialized with max values from specified data set.
     * @param dataSet
     */
    public MaxNormalizer(DataSet dataSet) {
        findMaxVectors(dataSet);
        intialized = true;
    }

    @Override
   public void normalize(DataSet dataSet) {

        for (DataSetRow row : dataSet.getRows()) {
            double[] normalizedInput = normalizeMax(row.getInput(), maxIn);
            row.setInput(normalizedInput);

            if (dataSet.isSupervised()) {
                double[] normalizedOutput = normalizeMax(row.getDesiredOutput(), maxOut);
                row.setDesiredOutput(normalizedOutput);
            }
        }
    }


   /**
    * Finds max values for columns in input and output vector for given data set
    * @param dataSet
    */
    private void findMaxVectors(DataSet dataSet) {
        int inputSize = dataSet.getInputSize();
        int outputSize = dataSet.getOutputSize();

        maxIn = new double[inputSize];
        for(int i=0; i<inputSize; i++) {
            maxIn[i] = Double.MIN_VALUE;
        }

        maxOut = new double[outputSize];
        for(int i=0; i<outputSize; i++)
            maxOut[i] = Double.MIN_VALUE;

        for (DataSetRow dataSetRow : dataSet.getRows()) {
            double[] input = dataSetRow.getInput();
            for (int i = 0; i < inputSize; i++) {
                if (Math.abs(input[i]) > maxIn[i]) {
                    maxIn[i] = Math.abs(input[i]);
                }
             }

            double[] output = dataSetRow.getDesiredOutput();
            for (int i = 0; i < outputSize; i++) {
                if (Math.abs(output[i]) > maxOut[i]) {
                    maxOut[i] = Math.abs(output[i]);
                }
            }
        }
    }

    private double[] normalizeMax(double[] vector, double[] max) {
        double[] normalizedVector = new double[vector.length];

        for(int i = 0; i < vector.length; i++) {
                normalizedVector[i] = vector[i] / max[i];
        }

        return normalizedVector;
    }

}
