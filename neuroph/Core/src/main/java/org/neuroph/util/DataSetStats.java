package org.neuroph.util;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

/**
 * Utility class with methods for calculating dataset statistics
 * TODO: add variance, and std
 * Calculate everything in one pass and expose as attributes - like summary() in R
 * Not only for inputs but also for outputs
 */
public class DataSetStats {

    /**
     *
     * @param dataSet Neuroph dataset
     * @return arithmetic mean for each input in data set
     */
	public static double[] inputsMean(DataSet dataSet) {
		double[] mean = new double[dataSet.getInputSize()];

		for (DataSetRow row : dataSet.getRows()) {
			double[] currentInput = row.getInput();
			for (int i = 0; i < dataSet.getInputSize(); i++) {
				mean[i] += currentInput[i];
			}
		}
		for (int i = 0; i < dataSet.getInputSize(); i++) {
			mean[i] /= (double)dataSet.getRows().size();
		}
		return mean;
	}

    /**
     *
     * @param dataSet Neuroph dataset
     * @return maximum value for each input in data set
     */
	public static double[] inputsMax(DataSet dataSet) {

		int inputSize = dataSet.getInputSize();
		double[] maxColumnElements = new double[inputSize];

		for (int i = 0; i < inputSize; i++) {
			maxColumnElements[i] = -Double.MAX_VALUE;
		}

		for (DataSetRow dataSetRow : dataSet.getRows()) {
			double[] input = dataSetRow.getInput();
			for (int i = 0; i < inputSize; i++) {
				maxColumnElements[i] = Math.max(maxColumnElements[i], input[i]);
			}
		}

		return maxColumnElements;
	}

    /**
     *
     * @param dataSet Neuroph dataset
     * @return minimum value for each variable in data set
     */
	public static double[] inputsMin(DataSet dataSet) {

		int inputSize = dataSet.getInputSize();
		double[] minColumnElements = new double[inputSize];

		for (int i = 0; i < inputSize; i++) {
			minColumnElements[i] = Double.MAX_VALUE;
		}

		for (DataSetRow dataSetRow : dataSet.getRows()) {
			double[] input = dataSetRow.getInput();
			for (int i = 0; i < inputSize; i++) {
				minColumnElements[i] = Math.min(minColumnElements[i], input[i]);
			}
		}
		return minColumnElements;
	}

}
