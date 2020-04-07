package org.neuroph.core.data;

import org.neuroph.util.data.norm.MaxNormalizer;

/**
 * Utility methods for dowrking with datasets
 */
public class DataSets {

    public static MaxNormalizer normalizeMax(DataSet dataSet) {
        MaxNormalizer maxNorm = new MaxNormalizer(dataSet);
        maxNorm.normalize(dataSet);
        return maxNorm;
    }

    public static DataSet[] trainTestSplit(DataSet dataSet, double split){
        DataSet[] trainTestSet =  dataSet.split(split, 1-split);
        return trainTestSet;
    }

    public static DataSet readFromCsv(String filePath, int inputsCount, int outputsCount, String delimiter) {
        return DataSet.createFromFile(filePath, inputsCount, outputsCount, delimiter);
    }

    public static DataSet readFromCsv(String filePath, int inputsCount, int outputsCount) {
        return DataSet.createFromFile(filePath, inputsCount, outputsCount, ",");
    }
}
