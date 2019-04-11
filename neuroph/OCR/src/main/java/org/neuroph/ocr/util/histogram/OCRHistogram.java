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
 *
 * @author Mihailo
 */
public class OCRHistogram {


    /**
     * Calculate the width histogram for single row. <br/>
     * Make the rectangle with: <br/>
     * width = width of the image<br/>
     * height = predicted height of letter<br/>
     * It scans this rectangle by width, start from left to right and finds all black
     * pixels in each column. Method returns array which length is width of the image.
     * Every element in array corresponds to number of black pixels in the column of
     * the rectangle.
     * @param image input image with multiple lines and letters
     * @param row pixel position of the row. It should be center of the single row.
     * This number can be calculated by method findRowPixels
     * @param letterHeight predicted letter size (above and below row)
     * @return
     */
    public static int[] widthRowHistogram(BufferedImage image, int row, int letterHeight) {
        int width = image.getWidth();
        int height = image.getHeight();
        int color;
        int black = 0;
        int[] histogram = new int[width];
        for (int i = 0; i < width; i++) {
            for (int j = row - (letterHeight / 2); j <= row + (letterHeight / 2); j++) {
                if (j < 0 || j >= height) {
                    continue;
                }
                color = new Color(image.getRGB(i, j)).getRed();
                if (color == black ) {
                    histogram[i]++;
                }
            }

        }
        return histogram;
    }

     /**
     * Method for finding histogram but with ignoring spaces between lines/words<br/>
     * When you use only widthHistogram() method, it will find spaces between letters.
     * With this method you need to set the space gap will be ignored so you will get new
     * histogram where the words are separated, not letters.
     * @param histogram classic width histogram
     * @param spaceGap size of the space which will be ignored
     * @return new histogram with ignored spaces. <br/>
     * Previously in histogram these spaces have had values of zero<br/>
     * Now they are filled with ones: histogram[i] = 1
     */
    public static int[] histogramWithoutLetterSpaces(int[] histogram, int spaceGap) {
        int count = 0;
        for (int i = 0; i < histogram.length; i++) {
            if (histogram[i] == 0) {
                count++;
            } else { //(histogram[i] != 0) drugim recima vece je od nule
//                System.out.println(i+"-"+count);
                if (count > 0 && count < spaceGap) {
                    for (int j = i - count; j < i; j++) {
                        histogram[j] = 1; // letter space
                    }
                }
                count = 0;
            }

        }
        return histogram;
    }

}
