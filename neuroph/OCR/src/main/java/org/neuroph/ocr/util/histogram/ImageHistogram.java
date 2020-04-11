/**
 * Copyright 2014 Neuroph Project http://neuroph.sourceforge.net
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
package org.neuroph.ocr.util.histogram;

import java.awt.Color;
import java.awt.image.BufferedImage;

/**
 * Utility methods to calculate image histograms by height and width, and gradient.
 *
 * @author Mihailo
 */
public class ImageHistogram {
    public static final int BLACK = 0;
    
    
    /**
     * Create and return histogram for binarized input image by height.
     * Histogram contains number of black pixels in image pixel rows.
     * Returns array which length is height of image, every element of array
     * represents count of black pixels in that row.
     *
     * @param image binarized image, letters are black, background is white
     * @return image histogram
     */
    public static int[] histogramByHeight(final BufferedImage image) {
        final int height = image.getHeight();
        final int width = image.getWidth();
        final int[] histogram = new int[height];
        
        int color;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                color = new Color(image.getRGB(x, y)).getRed();
                if (color == BLACK) {
                    histogram[y]++;
                }
            }
        }
        return histogram;
    }


    /**
     * @param image binarized image, letters are black, background is white
     * @return array which length is width of image, every element of array
     * represent count of black pixels in that column of pixels.
     */
    public static int[] histogramByWidth(BufferedImage image) {
        final int height = image.getHeight();
        final int width = image.getWidth();
        final int[] histogram = new int[width];

        int color;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                color = new Color(image.getRGB(x, y)).getRed();
                if (color == BLACK) {
                    histogram[x]++;
                }
            }
        }
        return histogram;
    }

    /**
     * @param histogram histogram calculated by method
     * <b>heightHistogram(BufferedImage)</b> or
     * <b>widthHistogram(BufferedImage)</b>
     * Each element in array is calculated in the following way:<br/>
     * gradient[i] = histogram[i] - histogram[i-1]
     * 
     * @return array that represents gradient 
     */
    public static int[] gradient(int[] histogram) {
        int[] gradient = new int[histogram.length];
        for (int i = 1; i < gradient.length; i++) {
            gradient[i] = histogram[i] - histogram[i - 1];
        }
        return gradient;
    }

}
