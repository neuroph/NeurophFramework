package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 * Dilation filter is used for making lines on the image little bit wider. It convolves through whole image
 * and every black pixel replaces with 9 pixels. 
 * 
 * If at least one pixel in the structuring element coincides with a foreground pixel in the image underneath, then the input pixel is set to the foreground value.
 * Most implementations of this operator expect the input image to be binary, usually with foreground pixels at pixel value 255, and background pixels at pixel value 0.
 * 
 * https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm
 * 
 * @author Mihailo Stupar
 */
public class Dilation implements ImageFilter<BufferedImage>,Serializable{
     
    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;
    private transient int width;
    private transient int height;
    public static final int WHITE = 255;
    public static final int BLACK = 0;    
    private final int radius;    
    
    public Dilation() {
        this.radius = 2;
    }        
    
    public Dilation(int radius) {
        this.radius = radius;
    }    
    
    @Override
    public BufferedImage apply(BufferedImage image) {        
        originalImage = image;        
        width = originalImage.getWidth();
        height = originalImage.getHeight();        
        filteredImage = new BufferedImage(width, height, originalImage.getType());
        
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int color = new Color(originalImage.getRGB(x, y)).getRed();
                if (color == BLACK) { // ako je boja bixela crna primeni filter na toj poziciji
                    convolve(x, y);
                } else { // ako nije nek bude bela
                    int alpha = new Color(originalImage.getRGB(x, y)).getAlpha();
                    int rgb = ImageUtilities.argbToColor(alpha, WHITE, WHITE, WHITE);
                    filteredImage.setRGB(x, y, rgb);
                }         
            }
        }
        return filteredImage;
    }
        
    // applies convolution operator at the specified position
    private void convolve (int xCenter, int yCenter) {
        for (int x = xCenter-radius; x <= xCenter+radius; x++) {
            for (int y = yCenter-radius; y <= yCenter+radius; y++) {
                if (x>=0 && y>=0 && x<width && y<height) {
                    int alpha = new Color(originalImage.getRGB(x, y)).getAlpha();
                    int rgb = ImageUtilities.argbToColor(alpha, BLACK, BLACK, BLACK);
                    filteredImage.setRGB(x, y, rgb);
                }
            }
        }
    }

    @Override
    public String toString() {
        return "Dilation";
    }

}