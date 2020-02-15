package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 *
 * @author Mihailo Stupar
 *
 * Improves the quality of handwriting letters.
 */
public class NormalizationFilter implements ImageFilter<BufferedImage> {

    private BufferedImage originalImage;
    private BufferedImage filteredImage;

    private int blockSize = 5; //should be odd number (ex. 5)

    private double targetMean = 0;
    private double targetVariance = 1;

    private int mean;
    private int var;

    private int width;
    private int height;

    private int[][] imageMatrix;

    @Override

    public BufferedImage apply(BufferedImage image) {

        originalImage = image;

        width = originalImage.getWidth();
        height = originalImage.getHeight();

        filteredImage = new BufferedImage(width, height, originalImage.getType());
        imageMatrix = new int[width][height];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {

                imageMatrix[i][j] = new Color(originalImage.getRGB(i, j)).getRed();

            }
        }

        mean = calculateMean();
        var = calculateVariance();

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {

                double normalizedPixel = 0;
                double squareError = 0;

                if (imageMatrix[i][j] > mean) {
                    squareError = (imageMatrix[i][j] - mean) * (imageMatrix[i][j] - mean);
                    normalizedPixel = (targetMean + Math.sqrt(((targetVariance * squareError / var))));
                } else {
                    squareError = (imageMatrix[i][j] - mean) * (imageMatrix[i][j] - mean);
                    normalizedPixel = (targetMean - Math.sqrt(((targetVariance * squareError / var))));
                }

                int alpha = new Color(originalImage.getRGB(i, j)).getAlpha();
                int rgb = (int) -normalizedPixel;
                int color = ImageUtilities.argbToColor(alpha, rgb, rgb, rgb);
                filteredImage.setRGB(i, j, color);
            }
        }

        return filteredImage;
    }

    public int calculateVariance() {

        int var = 0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                var += (imageMatrix[x][y] - mean) * (imageMatrix[x][y] - mean);
            }
        }
        return (int) var / (height * width * 255); //255 for white color
    }

    public int calculateMean() {
        double mean = 0;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                mean += imageMatrix[x][y];
            }
        }

        return (int) mean / (width * height);
    }

    @Override
    public String toString() {
        return "Normalization Filter";
    }

    public void setTargetMean(double targetMean) {
        this.targetMean = targetMean;
    }

    public void setTargetVariance(double targetVariance) {
        this.targetVariance = targetVariance;
    }

}
