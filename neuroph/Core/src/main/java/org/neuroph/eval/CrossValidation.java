/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.neuroph.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.util.data.sample.SubSampling;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.learning.error.MeanSquaredError;


public class CrossValidation {

    Evaluation evaluation;
    NeuralNetwork neuralNet;
    DataSet dataSet;
    EvaluationResult totalResult;
    int numfolds;
    
    
    List<ConfusionMatrix> cmlist;
    List<ClassificationMetrics.Stats> statlist;
    List<CrossFolds> crosslist;
    
   

    public CrossValidation(NeuralNetwork neuralnet, DataSet dataset, int numFolds) {
        
        neuralNet = neuralnet;
        dataSet = dataset;
        numfolds = numFolds;
        
    }
    
   

    public EvaluationResult run() throws InterruptedException, ExecutionException {

         int sizePercent = 100 / numfolds;
        int numberargs = 100 / sizePercent;

        int[] varint = new int[numberargs];
        for (int i = 0; i <= numberargs - 1; i++) {
            if (numberargs - 1 == i) {
                int sizelast = 100 - (numberargs - 1) * sizePercent;
                varint[i] = sizelast;
            } else {
                varint[i] = sizePercent;
            }
        }

        SubSampling s = new SubSampling(varint);
        List<DataSet> foldSetList = s.sample(dataSet);
        

        cmlist = new ArrayList<>();
        statlist = new ArrayList<>();
        crosslist=new ArrayList<>();
        
        ArrayList<CrossValidationWorker> workersTasks = new ArrayList<>();
        
        for (DataSet validationSet : foldSetList) {
           DataSet learningSet=returnSet(dataSet,validationSet);
           CrossValidationWorker cw= new CrossValidationWorker(neuralNet,learningSet,validationSet);
           workersTasks.add(cw);
        }
        
        ExecutorService executor = Executors.newFixedThreadPool(4);
        List<Future<CrossFolds>> evaluationResults = executor.invokeAll(workersTasks);
        executor.shutdown();
        
        List<CrossFolds> results=new ArrayList<>();

        for (Future<CrossFolds> evaluationResult : evaluationResults) {
            results.add(evaluationResult.get());
        }

         int foldNumber=1;
        for (CrossFolds crossfolds: results){
         showResults(crossfolds.getConfusionMatrix(),foldNumber);
         cmlist.add(crossfolds.getConfusionMatrix());
         crosslist.add(crossfolds);
         ClassificationMetrics.Stats average = ClassificationMetrics.average(ClassificationMetrics.createFromMatrix(crossfolds.getConfusionMatrix()));
          statlist.add(average);
          foldNumber++;}
           
        ConfusionMatrix sumMatrix = sumConfusionMatrix(cmlist, dataSet);

        System.out.println();
        System.out.println("All folds results:");
        System.out.println();

        EvaluationResult sumEval = new EvaluationResult();
        sumEval.setConfusionMatrix(sumMatrix);
        showResults(sumMatrix,0);
        

       ClassificationMetrics.Stats allstat = averageStats(statlist);
       showResults(dataSet, allstat, numfolds);
       
       return sumEval;
    }

    public ConfusionMatrix sumConfusionMatrix(List<ConfusionMatrix> cmlist, DataSet dataset) {
 
        ConfusionMatrix cm = new ConfusionMatrix(cmlist.get(0).getClassLabels());
        int[][] ar = new int[dataset.getOutputSize()][dataset.getOutputSize()];
        for (ConfusionMatrix c : cmlist) {

            for (int i = 0; i < dataset.getOutputSize(); ++i) {

                for (int j = 0; j < dataset.getOutputSize(); ++j) {
                    ar[i][j] += c.get(i, j);
                }
            }
        }

        cm.setValues(ar);
        return cm;
    }
    
    public static ClassificationMetrics.Stats averageStats(List<ClassificationMetrics.Stats> metricsList) {
       
         ClassificationMetrics.Stats average = metricsList.get(0);
          double count = 1;
            for (ClassificationMetrics.Stats st : metricsList) {
                if (!st.equals(average)){
                average.accuracy += st.accuracy;
                average.precision += st.precision;
                average.recall += st.recall;
                average.fScore += st.fScore;}
            }
            count++;
        
        count = metricsList.size(); 
        average.accuracy = average.accuracy / count;
        average.precision = average.precision / count;
        average.recall = average.recall / count;
        average.fScore = average.fScore / count;
       
        return average;
    }

    public void showResults(DataSet dataset, ClassificationMetrics.Stats nst, int numfolds) {
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

    public void showResults(ConfusionMatrix confusionMatrix, int foldIndex){
     
        System.out.println();
            System.out.println("Fold " + foldIndex);
            System.out.println();
        System.out.println("Confusion matrrix:\r\n");
        System.out.println(confusionMatrix.toString() + "\r\n\r\n");
        System.out.println("Classification metrics\r\n");
        ClassificationMetrics[]  metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
         ClassificationMetrics.Stats stat = ClassificationMetrics.average(metrics);
        for (ClassificationMetrics cm : metrics) {
           System.out.println(cm.toString() + "\r\n");
        }
        
       System.out.println(stat.toString());
    
    }
    
     public void showStats(ConfusionMatrix confusionMatrix){
     ClassificationMetrics[]  metrics = ClassificationMetrics.createFromMatrix(confusionMatrix);
     ClassificationMetrics.Stats stat = ClassificationMetrics.average(metrics);
     System.out.println(stat.toString());
    }
     
     public DataSet returnSet(DataSet dataSet, DataSet validationSet){
      DataSet learningSet= new DataSet(dataSet.getInputSize(),dataSet.getOutputSize());
      learningSet.addAll(dataSet);
     learningSet.removeAll(validationSet);
     return learningSet;
     }
     
     public class CrossValidationWorker implements Callable<CrossFolds> {

        private final DataSet learningSet;
        private final DataSet trainingSet;

         
         
         public CrossValidationWorker(NeuralNetwork neuralNetwork, DataSet learningSet, DataSet validationSet) {
           this.trainingSet=validationSet;
            this.learningSet=learningSet;
        }
         
        
        @Override
        public CrossFolds call() throws Exception {
            Evaluation evaluation = new Evaluation();
            evaluation.addEvaluator(new ErrorEvaluator(new MeanSquaredError()));
            
            evaluation.addEvaluator(new ClassifierEvaluator.MultiClass(neuralNet.getOutputLabels()));
           
            neuralNet.learn(learningSet);
            

            EvaluationResult evaluationResult = evaluation.evaluateDataSet(neuralNet, trainingSet);
            CrossFolds crossFolds= new CrossFolds(evaluationResult.getNeuralNetwork(),learningSet,trainingSet);
           crossFolds.setConfusionMatrix(evaluationResult.getConfusionMatrix());
          
           return crossFolds;
        }
     }
     
     public  void setFoldResults(List<CrossFolds> crosslist){
     this.crosslist=crosslist;
     }
     
     public List<CrossFolds> getFoldResults(){
     return crosslist;
     }
}
