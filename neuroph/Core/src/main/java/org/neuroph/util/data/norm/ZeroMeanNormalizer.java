package org.neuroph.util.data.norm;

import java.io.Serializable;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.util.DataSetStatistics;

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
        DataSetStatistics stats = new DataSetStatistics(dataSet);
        maxInput = stats.inputsMax(); // znaci ovde se koristi ta klasa
        minInput = stats.inputsMin();
        meanInput = stats.inputsMean();
        stdInput =  stats.inputsStandardDeviation(meanInput);
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