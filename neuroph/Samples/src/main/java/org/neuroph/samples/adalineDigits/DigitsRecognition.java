package org.neuroph.samples.adalineDigits;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;

/**
 * In this sample MultiLayerPerceptron network is used for pattern recognition.  The
 * input pattern must match EXACTLY with what the network was trained with.
 *
 * This example trains Adaline network to recognize the 10 digits.
 *
 * This is based on a an example from Encog (Encog Examples/org.encog.examples.neural.adaline).
 * Encog example is based on a an example by Karsten Kutza, written in C on 1996-01-24. http://www.neural-networks-at-your-fingertips.com
 *
 * @author Ivan Petrovic
 */
public class DigitsRecognition {

    public static void main(String args[]) {

        //create training set from Data.DIGITS
        DataSet dataSet = generateTrainingSet();

        int inputCount = DigitData.CHAR_HEIGHT * DigitData.CHAR_WIDTH;
        int outputCount = DigitData.DIGITS.length;
        int hiddenNeurons = 19;

        //create neural network
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputCount, hiddenNeurons, outputCount);
        //get backpropagation learning rule from network
        BackPropagation learningRule = neuralNet.getLearningRule();

        learningRule.setLearningRate(0.5);
        learningRule.setMaxError(0.001);
        learningRule.setMaxIterations(5000);

        //add learning listener in order to print out training info
        learningRule.addListener(new LearningEventListener() {
            @Override
            public void handleLearningEvent(LearningEvent event) {
                BackPropagation bp = (BackPropagation) event.getSource();
                if (event.getEventType().equals(LearningEvent.Type.LEARNING_STOPPED)) {
                    System.out.println();
                    System.out.println("Training completed in " + bp.getCurrentIteration() + " iterations");
                    System.out.println("With total error " + bp.getTotalNetworkError() + '\n');
                } else {
                    System.out.println("Iteration: " + bp.getCurrentIteration() + " | Network error: " + bp.getTotalNetworkError());
                }
            }
        });

        //train neural network
        neuralNet.learn(dataSet);

        //train the network with training set
        testNeuralNetwork(neuralNet, dataSet);

    }

    /**
     * Prints network output for the each element from the specified training
     * set.
     *
     * @param neuralNet neural network
     * @param testSet test data set
     */
    public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        System.out.println("--------------------------------------------------------------------");
        System.out.println("***********************TESTING NEURAL NETWORK***********************");
        for (DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();

            int outputIdx = maxOutput(neuralNet.getOutput());

            String[] inputDigit = DigitData.convertDataIntoImage(testSetRow.getInput());

            for (int i = 0; i < inputDigit.length; i++) {
                if (i != inputDigit.length - 1) {
                    System.out.println(inputDigit[i]);
                } else {
                    System.out.println(inputDigit[i] + "----> " + outputIdx);
                }
            }
            System.out.println("");
        }
    }

    /**
     * Creates and returns training data as instance of DataSet
     * @return  training data as DataSet instance
     */
    public static DataSet generateTrainingSet() {

        DataSet dataSet = new DataSet(DigitData.CHAR_WIDTH * DigitData.CHAR_HEIGHT, DigitData.DIGITS.length);

        for (int i = 0; i < DigitData.DIGITS.length; i++) {
            // setup input
            DataSetRow inputRow = DigitData.convertImageIntoData(DigitData.DIGITS[i]);
            double[] input = inputRow.getInput();

            // setup output
            double[] output = new double[DigitData.DIGITS.length];

            for (int j = 0; j < DigitData.DIGITS.length; j++) {
                if (j == i) {
                    output[j] = 1;
                } else {
                    output[j] = 0;
                }
            }
            //creating new training element with specified input and output
            DataSetRow row = new DataSetRow(input, output);
            //adding row to data set
            dataSet.add(row);
        }
        return dataSet;
    }

    /**
     * Returns index of max element in given array.
     * @param array array to search for max
     * @return index of max element
     */
    public static int maxOutput(double[] array) {
        double max = array[0];
        int index = 0;

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                index = i;
                max = array[i];
            }
        }
        return index;
    }

}
