/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.neuroph.contrib.rnn.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.jblas.DoubleMatrix;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class SequenceModeller {

    private final Map<String, Integer> charIndex = new HashMap<>();
    private final Map<Integer, String> indexChar = new HashMap<>();
    private final Map<String, DoubleMatrix> charVector = new HashMap<>();
    private final List<String> sequence = new ArrayList<>();

    public SequenceModeller(DataSet dataSet) {
        loadData(dataSet);
        buildDistributedRepresentations();
    }

    private void loadData(DataSet dataSet) {
        List<DataSetRow> dataSetRows = dataSet.getRows();
        dataSetRows.forEach(row -> {
            StringBuilder stringBuilder = new StringBuilder();

            for (double desiredOutput : row.getDesiredOutput()) {
                stringBuilder.append(desiredOutput);
                stringBuilder.append("");
            }

            String rowString = stringBuilder.toString();
            sequence.add(rowString);

            for (char c : rowString.toLowerCase().toCharArray()) {
                String key = String.valueOf(c);
                if (!charIndex.containsKey(key)) {
                    charIndex.put(key, charIndex.size());
                    indexChar.put(charIndex.get(key), key);
                }
            }
        });
    }

    private void buildDistributedRepresentations() {
        for (String key : charIndex.keySet()) {
            DoubleMatrix timestepInput = DoubleMatrix.zeros(1, charIndex.size());
            timestepInput.put(charIndex.get(key), 1);
            charVector.put(key, timestepInput);
        }
    }

    public Map<String, Integer> getCharIndex() {
        return charIndex;
    }

    public Map<String, DoubleMatrix> getCharVector() {
        return charVector;
    }

    public List<String> getSequence() {
        return sequence;
    }

    public Map<Integer, String> getIndexChar() {
        return indexChar;
    }

}
