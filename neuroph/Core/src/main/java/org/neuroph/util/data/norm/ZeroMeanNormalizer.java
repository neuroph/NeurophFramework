package org.neuroph.util.data.norm;

import java.io.Serializable;
import org.neuroph.util.DataSetStats;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

/**
 * Normalizes data sets by shifting all values in such way that data set has mean of 0 and standard deviation 1 (aka standardization).
 */
public class ZeroMeanNormalizer implements Normalizer, Serializable {

    private double[] maxInput;
    private double[] minInput;
    private double[] meanInput;
    private double[] stdInput; // standard deviation
    // what about outputs?

    public ZeroMeanNormalizer(DataSet dataSet) {
        maxInput = DataSetStats.inputsMax(dataSet); // znaci ovde se koristi ta klasa
        minInput = DataSetStats.inputsMin(dataSet);
        meanInput = DataSetStats.inputsMean(dataSet);
        stdInput =  DataSetStats.inputsStandardDeviation(dataSet, meanInput);
    }

    @Override
    public void normalize(DataSet dataSet) {

        for (DataSetRow row : dataSet.getRows()) {
            double[] normalizedInput = row.getInput();

            for (int i = 0; i < dataSet.getInputSize(); i++) {
                normalizedInput[i] = (normalizedInput[i] - meanInput[i]) / stdInput[i];
            }
            row.setInput(normalizedInput);
        }
    }
}