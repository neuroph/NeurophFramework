package org.neuroph.eval;

import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;

/**
 * Result from single cross-validation fold, includes neural network, training and validation set,
 * and fold evaluation results (at the moment only confsionMatrix)
 *
 * TODO: add different eveluation metrics, for regression too.
 * 
 * @author Nevena Mlenkovic
 */
public class FoldResult {

    private final NeuralNetwork neuralNet;
    private final DataSet trainingSet;
    private final DataSet validationSet;
    private ConfusionMatrix confusionMatrix;
    // what if this is not classification but regression ? missing evaluation metrics

    public FoldResult(NeuralNetwork neuralNet, DataSet trainingSet, DataSet validationSet) {
        this.neuralNet = neuralNet;
        this.trainingSet = trainingSet;
        this.validationSet = validationSet;
    }

    /**
     * Returns neural network trained in this cross-validation fold.
     *
     * @return the neuralNet
     */
    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    /**
     * @return the trainingSet
     */
    public DataSet getTrainingSet() {
        return trainingSet;
    }

    /**
     * @return the validationSet
     */
    public DataSet getValidationSet() {
        return validationSet;
    }

    /**
     * @return the confusionMatrix
     */
    public ConfusionMatrix getConfusionMatrix() {
        return confusionMatrix;
    }

    /**
     * @param confusionMatrix the confusionMatrix to set
     */
    public void setConfusionMatrix(ConfusionMatrix confusionMatrix) {
        this.confusionMatrix = confusionMatrix;
    }

}