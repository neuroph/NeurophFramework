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
public class LossFunction {

    private static double getCategoricalCrossEntropy(DoubleMatrix p, DoubleMatrix q) {
        for (int i = 0; i < q.length; i++) {
            if (q.get(i) == 0) {
                q.put(i, 1e-10);
            }
        }
        return -p.mul(MatrixFunctions.log(q)).sum();
    }

    public static double getMeanCategoricalCrossEntropy(DoubleMatrix P, DoubleMatrix Q) {
        double e = 0;
        if (P.rows == Q.rows) {
            for (int i = 0; i < P.rows; i++) {
                e += getCategoricalCrossEntropy(P.getRow(i), Q.getRow(i));
            }
            e /= P.rows;
        } else {
            System.exit(-1);
        }
        return e;
    }

}
