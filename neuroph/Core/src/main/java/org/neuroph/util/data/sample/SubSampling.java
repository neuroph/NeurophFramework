/**
 * Copyright 2013 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.neuroph.util.data.sample;

import java.util.ArrayList;
import java.util.List;

import org.neuroph.core.data.DataSet;

/**
 * This class provides sub-sampling of a data set, and creates a specified number of subsets form given data set.
 *
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class SubSampling implements Sampling {

    /**
     * Number of sub sets
     */
    private int numSubSets;

    /**
     * Sizes of each subset as decimal numbers with sum equal to 1.
     */
    private double[] subSetSizes;


    /**
     * Sampling will produce a specified number of subsets of equal sizes.
     * Handy for sub-sampling in KFold crossvalidation
     *
     * @param numSubSets number of subsets to create
     */
    public SubSampling(int numSubSets) {
        this.numSubSets = numSubSets;
        this.subSetSizes = null; // now known util specific data set is given
    }


    /**
     * Sampling will create subsets of specified sizes.
     * Sum of specified sub set sizes must be 1.
     *
     * @param subSetSizes size of subsets in percents
     */
    public SubSampling(double ... subSetSizes) { // we should use doubles here
        double sum=0;
        for(int i=0; i<subSetSizes.length; i++) {
            sum+=subSetSizes[i];
        }
        if (sum>1) throw new IllegalArgumentException("Sum of sub set sizes cannot be greater then 1");

         this.subSetSizes = subSetSizes;
         this.numSubSets = subSetSizes.length;
    }

    @Override
    public DataSet[] sample(DataSet dataSet) {
        // if object was initializes by specifying numParts calculate subSetSizes so all subsets are equally sized
        if (subSetSizes == null) {
            final double singleSubSetSize = 1.0d / numSubSets;
            subSetSizes = new double[numSubSets];
            for (int i = 0; i < numSubSets; i++) {
                subSetSizes[i] = singleSubSetSize;
            }
        }

        // create list of data sets to return
        List<DataSet> subSets = new ArrayList<>();

        // shuffle dataset in order to randomize rows that will be used to fill subsets
        dataSet.shuffle();

        int idxCounter = 0; // index of main data set
        for (int subSetIdx = 0; subSetIdx < numSubSets; subSetIdx++) {
            // create new subset
            DataSet newSubSet = new DataSet(dataSet.getInputSize(), dataSet.getOutputSize());
            // cop column names if there are any
            newSubSet.setColumnNames(dataSet.getColumnNames());

            // fill subset with rows
            long subSetSize = Math.round(subSetSizes[subSetIdx] * dataSet.size()); // calculate size of the current subset
            for (int i = 0; i < subSetSize; i++) {
                if (idxCounter >= dataSet.size()) {
                    break;
                }
                newSubSet.add(dataSet.getRowAt(idxCounter));
                idxCounter++;
            }

            // add current subset to list that will be returned
            subSets.add(newSubSet);
        }

        return subSets.toArray(new DataSet[numSubSets]);
    }
}