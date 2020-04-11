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

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class Activation {

    public static DoubleMatrix logistic(DoubleMatrix inputMatrix) {
        return MatrixFunctions.pow(MatrixFunctions.exp(inputMatrix.mul(-1)).add(1), -1);
    }

    public static DoubleMatrix tanh(DoubleMatrix inputMatrix) {
        return MatrixFunctions.tanh(inputMatrix);
    }

    public static DoubleMatrix softmax(DoubleMatrix inputMatrix) {
        DoubleMatrix exponentMatrix = MatrixFunctions.exp(inputMatrix);
        for (int i = 0; i < inputMatrix.rows; i++) {
            DoubleMatrix exponenetMatrixRow = exponentMatrix.getRow(i);
            exponentMatrix.putRow(i, exponenetMatrixRow.div(exponenetMatrixRow.sum()));
        }
        return exponentMatrix;
    }

}
