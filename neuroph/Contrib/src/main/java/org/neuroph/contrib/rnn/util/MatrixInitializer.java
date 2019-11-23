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

import java.util.Random;
import org.jblas.DoubleMatrix;

/**
 *
 * @author Milan Šuša <milan_susa@hotmail.com>
 */
public class MatrixInitializer {

    public enum Type {
        Uniform, Gaussian
    }

    private final Type type;
    private static final Random RANDOM = new Random();

    private double scale = 0.01;
    private double miu = 0;
    private double sigma = 0.01;

    public MatrixInitializer(Type type, double scale, double miu, double sigma) {
        this.type = type;
        this.scale = scale;
        this.miu = miu;
        this.sigma = sigma;
    }

    public DoubleMatrix uniform(int rows, int cols) {
        return DoubleMatrix.rand(rows, cols).mul(2 * scale).sub(scale);
    }

    public DoubleMatrix gaussian(int rows, int cols) {
        DoubleMatrix matrix = new DoubleMatrix(rows, cols);
        for (int i = 0; i < matrix.length; i++) {
            matrix.put(i, RANDOM.nextGaussian() * sigma + miu);
        }
        return matrix;
    }

    public Type getType() {
        return type;
    }

    public double getScale() {
        return scale;
    }

    public double getMiu() {
        return miu;
    }

    public double getSigma() {
        return sigma;
    }

}
