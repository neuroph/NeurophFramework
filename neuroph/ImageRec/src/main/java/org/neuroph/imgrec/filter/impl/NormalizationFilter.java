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

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                imageMatrix[x][y] = new Color(originalImage.getRGB(x, y)).getRed();
            }
        }

        mean = calculateMean();
        var = calculateVariance();

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double normalizedPixel = 0;
                double squareError = 0;

                if (imageMatrix[x][y] > mean) {
                    squareError = (imageMatrix[x][y] - mean) * (imageMatrix[x][y] - mean);
                    normalizedPixel = (targetMean + Math.sqrt(((targetVariance * squareError / var))));
                } else {
                    squareError = (imageMatrix[x][y] - mean) * (imageMatrix[x][y] - mean);
                    normalizedPixel = (targetMean - Math.sqrt(((targetVariance * squareError / var))));
                }

                int alpha = new Color(originalImage.getRGB(x, y)).getAlpha();
                int rgb = (int) -normalizedPixel;
                int color = ImageUtilities.argbToColor(alpha, rgb, rgb, rgb);
                filteredImage.setRGB(x, y, color);
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
