package org.neuroph.util.data.norm;

import java.io.Serializable;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

/**
 * Performs normalization of a data set inputs and outputs to specified range.
 * 
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class RangeNormalizer implements Normalizer, Serializable {
    private double lowLimit=0, highLimit=1;
    private double[] maxIn, maxOut; // contains max values for in and out columns
    private double[] minIn, minOut; // contains min values for in and out columns

    // Da li ovde fali Abs? low moze biti i negativan!
    
    public RangeNormalizer(double lowLimit, double highLimit) {
        this.lowLimit= lowLimit;
        this.highLimit = highLimit;
    }

    @Override
    public void normalize(DataSet dataSet) {

        findMaxAndMinVectors(dataSet);

        for (DataSetRow row : dataSet.getRows()) {
            double[] normalizedInput = normalizeToRange(row.getInput(), minIn, maxIn);
            row.setInput(normalizedInput);

            if (dataSet.isSupervised()) {
                double[] normalizedOutput = normalizeToRange(row.getDesiredOutput(), minOut, maxOut);
                row.setDesiredOutput(normalizedOutput);
            }
        }
    }

    private double[] normalizeToRange(double[] vector, double[] min, double[] max) {
        double[] normalizedVector = new double[vector.length];

        for (int i = 0; i < vector.length; i++) {
            normalizedVector[i] = ((vector[i] - min[i]) / (max[i] - min[i])) * (highLimit - lowLimit) + lowLimit ;
        }

        return normalizedVector;
    }



    /**
     * Find min and max values for each position in vectors.
     *
     * @param dataSet
     */
    private void findMaxAndMinVectors(DataSet dataSet) {
        int inputSize = dataSet.getInputSize();
        int outputSize = dataSet.getOutputSize();

        maxIn = new double[inputSize];
        minIn = new double[inputSize];

        for(int i=0; i<inputSize; i++) {
            maxIn[i] = Double.MIN_VALUE;
            minIn[i] = Double.MAX_VALUE;
        }

        maxOut = new double[outputSize];
        minOut = new double[outputSize];

        for(int i=0; i<outputSize; i++) {
            maxOut[i] = Double.MIN_VALUE;
            minOut[i] = Double.MAX_VALUE;
        }

        for (DataSetRow dataSetRow : dataSet.getRows()) {
            double[] input = dataSetRow.getInput();
            for (int i = 0; i < inputSize; i++) {
                if (input[i] > maxIn[i]) {
                    maxIn[i] = input[i];
                }
                if (input[i] < minIn[i]) {
                    minIn[i] = input[i];
                }
            }

            double[] output = dataSetRow.getDesiredOutput();
            for (int i = 0; i < outputSize; i++) {
                if (output[i] > maxOut[i]) {
                    maxOut[i] = output[i];
                }
                if (output[i] < minOut[i]) {
                    minOut[i] = output[i];
                }
            }

        }
    }

    public double getLowLimit() {
        return lowLimit;
    }

    public double getHighLimit() {
        return highLimit;
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
