package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 *
 * @author Mihailo Stupar
 */
public class GenericConvolution implements ImageFilter<BufferedImage> {

    private BufferedImage originalImage;
    private BufferedImage filteredImage;

    private double[][] kernel;
    private boolean normalize;

    public GenericConvolution(double[][] kernel) {
        this.kernel = kernel;
    }

    @Override
    public BufferedImage apply(BufferedImage image) {

        originalImage = image;
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();

        filteredImage = new BufferedImage(width, height, originalImage.getType());

        int radius = kernel.length / 2;

        if (normalize) {
            normalizeKernel();
        }

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                double result = convolve(x, y, radius);
                int gray = (int) Math.round(result);
                int alpha = new Color(originalImage.getRGB(x, y)).getAlpha();
                int rgb = ImageUtilities.argbToColor(alpha, gray, gray, gray);
                filteredImage.setRGB(x, y, rgb);
            }
        }

        return filteredImage;
    }

    protected double convolve(int xCenter, int yCenter, int radius) {
        double sum = 0;
        int kernelX = 0;
        for (int x = xCenter - radius; x <= xCenter + radius; x++) {
            int kernelY = 0;
            for (int y = yCenter - radius; y <= yCenter + radius; y++) {
                if (x >= 0 && x < originalImage.getWidth() && y > 0 && y < originalImage.getHeight()) {
                    int color = new Color(originalImage.getRGB(x, y)).getRed();
                    sum = sum + color * kernel[kernelX][kernelY];
                }
                kernelY++;
            }
            kernelX++;
        }

        return sum;
    }

    /*
    * Mak sure that kernel element sum is 1
     */
    private void normalizeKernel() {
        int kernelSum = 0;
        for (int i = 0; i < kernel.length; i++) {
            for (int j = 0; j < kernel.length; j++) {
                kernelSum += kernel[i][j];
            }
        }

        for (int i = 0; i < kernel.length; i++) {
            for (int j = 0; j < kernel.length; j++) {
                kernel[i][j] = kernel[i][j] / kernelSum;
            }
        }
    }

    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    public void setKernel(double[][] kernel) {
        if (kernel.length % 2 == 0) {
            throw new RuntimeException("Kernel cannot be even number!");
        }
        this.kernel = kernel;
    }

    @Override
    public String toString() {
        return "Generic convolution";
    }

}
