package org.neuroph.nnet.learning.knn;

import java.io.Serializable;
import java.util.List;
import org.neuroph.nnet.learning.kmeans.KVector;

/**
 * Finds K closest (most similar) vectors for given vector.
 * 
 * TODO: create KNNClassifier and KNNRegresor
 * 
 * @author Zoran Sevarac
 */
public class KNearestNeighbour implements Serializable {

    /**
     * Data set of KVectors.
     */
    private List<KVector> dataSet;
    // maybe replace this with DataSetRow? only use inputs as vectors.
    // how do you assign classes? or mean value for regression?

    public KNearestNeighbour(List<KVector> dataSet) {
        this.dataSet = dataSet;
    }

    /**
     * Returns k nearest neighbours from data set or the given input vector.
     *
     * @param vector
     * @param k
     * @return
     */
    public KVector[] getKNearestNeighbours(KVector vector, int k) {
        KVector[] nearestNeighbours = new KVector[k];

        // calculate distances from all vectors in the data set entire dataset
        for (KVector otherVector : dataSet) {
            otherVector.setDistanceFrom(vector);
            //double distance = vector.distanceFrom(otherVector); // replaced with method above
            //otherVector.distanceFrom(vector);
        }

        // find k nearest vectors - not fully implemented...
        for (int i = 0; i < k; i++) { // ind min k times
            int minIndex = i;
            KVector minVector = dataSet.get(i);
            double minDistance = minVector.getDistance();

            for (int j = i + 1; j < dataSet.size(); j++) {
                if (dataSet.get(j).getDistance() <= minDistance) {
                    minVector = dataSet.get(j);
                    minDistance = minVector.getDistance();
                    minIndex = j;
                }
            }

            // swap elements at i and minIndex positions
            KVector temp = dataSet.get(i);
            dataSet.set(i, dataSet.get(minIndex));
            dataSet.set(minIndex, temp);

            nearestNeighbours[i] = dataSet.get(i);
        }

        return nearestNeighbours;
    }

    public List<KVector> getDataSet() {
        return dataSet;
    }

}
