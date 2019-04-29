/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.neuroph.eval;

import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;

/**
 *
 * @author User
 */
public class CrossFolds {

    
    private NeuralNetwork neuralNet;
    private DataSet learningSet;
    private DataSet validationSet;
    private ConfusionMatrix confusionMatrix;
    
    public CrossFolds(NeuralNetwork neuralNet, DataSet learningSet, DataSet validationSet){
    this.neuralNet=neuralNet;
    this.learningSet=learningSet;
    this.validationSet=validationSet;
    }
    
    /**
     * @return the neuralNet
     */
    public NeuralNetwork getNeuralNet() {
        return neuralNet;
    }

    /**
     * @param neuralNet the neuralNet to set
     */
    public void setNeuralNet(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }


    /**
     * @return the learningSet
     */
    public DataSet getLearningSet() {
        return learningSet;
    }

    /**
     * @param learningSet the learningSet to set
     */
    public void setLearningSet(DataSet learningSet) {
        this.learningSet = learningSet;
    }

    /**
     * @return the validationSet
     */
    public DataSet getValidationSet() {
        return validationSet;
    }

    /**
     * @param validationSet the validationSet to set
     */
    public void setValidationSet(DataSet validationSet) {
        this.validationSet = validationSet;
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
