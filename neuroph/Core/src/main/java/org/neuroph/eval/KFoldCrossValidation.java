package org.neuroph.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.commons.lang3.SerializationUtils;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.error.MeanSquaredError;

/**
 * This class implements multi-threaded cross validation procedure.
 * Splits data set into k subsets (folds), trains the network with data from k-1 and tests with one subset
 * Repeats the procedure k times each time using different subset for testing.
 *
 * TODO:
 * check classifier evaluation, strange results. What about true negative for multi class?
 * Return mean and std for metrics
 * Which network to use as a final result? Best one?
 * Evaluator for classifiers and regressors
 *
 * @author Boris Fulurija
 * @author Lukic Sasa Multithreading
 * @author Zoran Sevarac
 * @author Igor Jovic
 * @author Nevena Milenkovic
 */
public class KFoldCrossValidation {
    //todo: random seed for spliting! can be fixed before
    private final NeuralNetwork neuralNetwork;
    private final DataSet dataSet;
    private final int numFolds;

    private CountDownLatch allFoldsCompleted;

    //private Evaluation evaluation; general evaluation which should be set externally
//    private EvaluationResult totalResult;

    private List<ConfusionMatrix> confusionMatrices;
    private List<ClassificationMetrics.Stats> statlist;
    private List<FoldResult> crossFoldResults;

    public KFoldCrossValidation(NeuralNetwork neuralNetwork, DataSet dataSet, int numFolds) {
        this.neuralNetwork = neuralNetwork;
        this.dataSet = dataSet;
        this.numFolds = numFolds;
    }

    public EvaluationResult run() throws InterruptedException, ExecutionException {
        confusionMatrices = new ArrayList<>();
        statlist = new ArrayList<>();
        crossFoldResults = new ArrayList<>();

        // split data set into specified number of folds
        DataSet[] foldDataSets = dataSet.split(numFolds);

        // sve sto mogu prebaci u thread da bi bilo brze!
        ArrayList<CrossValidationWorker> workersTasks = new ArrayList<>();
        allFoldsCompleted = new CountDownLatch(numFolds);

        // create training and validation set  and also worker tasks
        for(int i=0; i<numFolds; i++) {
            DataSet validationSet = foldDataSets[i];
            DataSet trainingSet = createTrainingSetFromFolds(foldDataSets, i);
            CrossValidationWorker cvWorker = new CrossValidationWorker(trainingSet, validationSet);
            workersTasks.add(cvWorker);
        }

        // start all crossvaldiation threads
        ExecutorService executor = Executors.newFixedThreadPool(4);
        List<Future<FoldResult>> evaluationResults = executor.invokeAll(workersTasks); //

        List<FoldResult> results = new ArrayList<>();

        for (Future<FoldResult> foldResult : evaluationResults) {
            results.add(foldResult.get());
        }

        // ovde stavi barijeru budi siguran da su svi threadovi zavrsili!!! -
        //debuguj ovde!
        allFoldsCompleted.await();

        executor.shutdown();
//ovi nisu bas tacni
        int foldNumber = 1; // used just for printing
        for (FoldResult crossfolds : results) {
            //printFoldResults(crossfolds.getConfusionMatrix(), foldNumber); // print stats for individual fold - mislim da ovi jos nisu istrenirani!!!!
            confusionMatrices.add(crossfolds.getConfusionMatrix());
            crossFoldResults.add(crossfolds);
            ClassificationMetrics.Stats average = ClassificationMetrics.average(ClassificationMetrics.createFromMatrix(crossfolds.getConfusionMatrix()));
            statlist.add(average);
            foldNumber++;
        }
// a ovi nsu losi
        ConfusionMatrix sumMatrix = sumConfusionMatrix(confusionMatrices, dataSet);

        System.out.println();
        System.out.println("All folds results:");
        System.out.println();

        EvaluationResult sumEval = new EvaluationResult();
        sumEval.setConfusionMatrix(sumMatrix);
        printFoldResults(sumMatrix, 0);

        ClassificationMetrics.Stats allstat = averageClassificationMetrics(statlist);
        printResults(dataSet, allstat, numFolds); // prints all results

        // TODO: return results which can be printed or used in caller dont  print them from here

        return sumEval;
    }

    public ConfusionMatrix sumConfusionMatrix(List<ConfusionMatrix> cmList, DataSet dataSet) {
        ConfusionMatrix cm = new ConfusionMatrix(cmList.get(0).getClassLabels());
        int[][] ar = new int[dataSet.getOutputSize()][dataSet.getOutputSize()];
        for (ConfusionMatrix c : cmList) {
            for (int i = 0; i < dataSet.getOutputSize(); ++i) {
                for (int j = 0; j < dataSet.getOutputSize(); ++j) {
                    ar[i][j] += c.get(i, j);
                }
            }
        }

        cm.setValues(ar);
        return cm;
    }

    public static ClassificationMetrics.Stats averageClassificationMetrics(List<ClassificationMetrics.Stats> metricsList) {

        ClassificationMetrics.Stats average = metricsList.get(0);
        double count = 1;
        for (ClassificationMetrics.Stats st : metricsList) {
            if (!st.equals(average)) { // cemu ova provera?
                average.accuracy += st.accuracy;
                average.precision += st.precision;
                average.recall += st.recall;
                average.fScore += st.fScore;
            }
        }
        count++;

        count = metricsList.size();
        average.accuracy = average.accuracy / count;
        average.precision = average.precision / count;
        average.recall = average.recall / count;
        average.fScore = average.fScore / count;

        return average;
    }

    public void printResults(DataSet dataset, ClassificationMetrics.Stats nst, int numfolds) {
        System.out.println();
        System.out.println("=== Cross validation result ===");
        System.out.println("Instances: " + dataset.size());
        System.out.println("Number of folds: " + numfolds);
        System.out.println("\n");
        System.out.println("=== Summary ===");
        System.out.println("Accuracy: " + nst.accuracy);
        System.out.println("Precision: " + nst.precision);
        System.out.println("Recall: " + nst.recall);
        System.out.println("FScore: " + nst.fScore);
        System.out.println("Correlation coefficient: " + nst.correlationCoefficient);
    }

    public void printFoldResults(ConfusionMatrix confusionMatrix, int foldIdx) {
        System.out.println();
        System.out.println("Fold: " + foldIdx);
        System.out.println();
        System.out.println("Confusion matrrix:\r\n");
        System.out.println(confusionMatrix.toString() + "\r\n\r\n");
        System.out.println("Classification metrics\r\n");
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats stat = ClassificationMetrics.average(metrics);
        for (ClassificationMetrics cm : metrics) {
            System.out.println(cm.toString() + "\r\n");
        }

        System.out.println(stat.toString());
    }

    public void printStats(ConfusionMatrix confusionMatrix) {
        ClassificationMetrics[] metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
        ClassificationMetrics.Stats stat = ClassificationMetrics.average(metrics);
        System.out.println(stat.toString());
    }

    /**
     * Creates and returns training set by merging all folds from the given list of folds
     * except fold at specified index excludeIdx, which will be used as validation data set.
     *
     * @param folds
     * @param excludeIdx
     * @return
     */
    private DataSet createTrainingSetFromFolds(DataSet[] folds, int excludeIdx) {
        DataSet trainingSet = new DataSet(dataSet.getInputSize(), dataSet.getOutputSize());

        for(int i=0; i< folds.length; i++) {
            if (i==excludeIdx) continue;
            trainingSet.addAll(folds[i]);
        }

        return trainingSet;
    }

    private class CrossValidationWorker implements Callable<FoldResult> {
        private final DataSet trainingSet;
        private final DataSet validationSet;

        public CrossValidationWorker(DataSet trainingSet, DataSet validationSet) {
            this.trainingSet = trainingSet;
            this.validationSet = validationSet;
        }

        @Override
        public FoldResult call() throws Exception {
            // make a cloned copy of neural network
            NeuralNetwork neuralNet = SerializationUtils.clone(neuralNetwork);
            Evaluation evaluation = new Evaluation();
            evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));
            if (neuralNetwork.getOutputsCount() == 1) {
                evaluation.addEvaluator(new ClassifierEvaluator.Binary(0.5)); // classification threshold 0.5
            } else {
                evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(dataSet.getColumnNames()));
            }

            neuralNetwork.learn(trainingSet);

            EvaluationResult evaluationResult = evaluation.evaluate(neuralNet, validationSet);
            FoldResult foldResult = new FoldResult(neuralNet, trainingSet, validationSet);
            foldResult.setConfusionMatrix(evaluationResult.getConfusionMatrix());
            // todo: get mean and std of evaluation resulst and diferentialte regression anc classification

            allFoldsCompleted.countDown();

            return foldResult;
        }
    }

    public List<FoldResult> getResultsByFolds() {
        return crossFoldResults;
    }
}