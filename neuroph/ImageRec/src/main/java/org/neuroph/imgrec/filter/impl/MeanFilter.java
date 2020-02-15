package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 * Mean filtering is method of smoothing images, reducing the amount of intensity variation between one pixel and the next.
 * It is often used to reduce noise in images (eliminating pixel values which are unrepresentative of their surroundings). 
 * 
 * http://homepages.inf.ed.ac.uk/rbf/HIPR2/mean.htm
 * 
 * @author Mihailo Stupar
 */
public class MeanFilter implements ImageFilter<BufferedImage>,Serializable{

    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;

    /**
     * Radius around pixel to calculate neighborhood  mean, typically should be 1 or 2 (to get 3 or 5 neighbourhood conv filter).
     * Central pixel can be greater.
     */
    private int radius;
    private transient int imageWidth;
    private transient int imageHeight;

    public MeanFilter() {
        this.radius = 1;
    }
    
    public MeanFilter(int radius) {
        this.radius = radius;
    }    
    
    @Override
    public BufferedImage apply(BufferedImage image) {
    
        originalImage = image;
        
        imageWidth = originalImage.getWidth();
        imageHeight = originalImage.getHeight();
        
        filteredImage = new BufferedImage(imageWidth, imageHeight, originalImage.getType());
        
        for (int x = 0; x < imageWidth; x++) {
            for (int y = 0; y < imageHeight; y++) {                
                int mean = findMeanPixelColor(x, y);
                int alpha = new Color(originalImage.getRGB(x, y)).getAlpha();                
                int rgb = ImageUtilities.argbToColor(alpha, mean, mean, mean);
                filteredImage.setRGB(x, y, rgb);                
            }
        }
                
        return filteredImage;                
    }
    
    /**
     * Calculates color mean for the given center position of the filter
     * @param centerX
     * @param centerY
     * @return 
     */
    private int findMeanPixelColor (int centerX, int centerY) {       
        double sum = 0;
        final int numPixels = (2*radius+1) * (2*radius+1);  
        for (int x = centerX-radius; x <= centerX+radius; x++) {
            for (int y = centerY-radius; y <= centerY+radius; y++) {              
                if (x > 0 && x < imageWidth && y > 0 && y< imageHeight) {                
                    int color = new Color(originalImage.getRGB(x, y)).getRed(); // why we use only red component here? we assume grayscale filter has been applied previously
                   // int color = originalImage.getRGB(x, y);  // trebalo bi koritiiti sve komponente
                    sum = sum + color;
                }   
            }
        }
        return (int) Math.round(sum/numPixels);      
    }

    @Override
    public String toString() {
        return "Mean Filter";
    }

    public void setRadius(int radius) {
        this.radius = radius;
    }

    public int getRadius() {
        return radius;
    }
    
    
}
