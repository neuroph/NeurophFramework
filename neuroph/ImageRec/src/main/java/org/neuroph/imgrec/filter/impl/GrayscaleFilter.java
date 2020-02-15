package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 * Grayscale filter from image in RGB format makes grayscale image in way that
 * for each pixel, using value of red, green and blue color, calculates new
 * value using formula: gray = 0.21*red + 0.71*green + 0.07*blue
 * Grayscale filter is commonly used as first filter in Filter Chain and on
 * grayscale image other filters are applied.
 *
 * @author Mihailo Stupar
 */
public class GrayscaleFilter implements ImageFilter<BufferedImage>, Serializable {

    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;

    @Override
    public BufferedImage apply(BufferedImage image) {
        originalImage = image;
        int alpha, red, green, blue, gray;

        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        filteredImage = new BufferedImage(width, height, originalImage.getType());

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                alpha = new Color(originalImage.getRGB(x, y)).getAlpha();
                red = new Color(originalImage.getRGB(x, y)).getRed();
                green = new Color(originalImage.getRGB(x, y)).getGreen();
                blue = new Color(originalImage.getRGB(x, y)).getBlue();
                gray = (int) (0.21 * red + 0.71 * green + 0.07 * blue);
                gray = ImageUtilities.argbToColor(alpha, gray, gray, gray);
                filteredImage.setRGB(x, y, gray);
            }
        }
        return filteredImage;
    }

    @Override
    public String toString() {
        return "Grayscale Filter";
    }
}
