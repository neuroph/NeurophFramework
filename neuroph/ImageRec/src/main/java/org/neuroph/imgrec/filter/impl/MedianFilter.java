package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.Arrays;
import javax.swing.JPanel;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 * Median filter is used for noise reduction on the grayscale image. The filter 
 * works on way that for each pixel in the image one window is set around it. 
 * Radius of the window by default is set to 4. Then all the values of the pixels 
 * belonging to the window are being sorted and values are used to calculate new 
 * value that represents the median. The value of that pixels in filtered image 
 * is replaced with one that is obtained as the median.
 * @param radius radius of the window
 * 
 * @author Mihailo Stupar
 */
public class MedianFilter implements ImageFilter<BufferedImage>, Serializable{

    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;
	
    /**
     * Radius around pixel to calculate neighborhood  mean, typically should be 1 or 2 (to get 3 or 5 neighbourhood conv filter).
     * Central pixel can be greater.
     */    
    private int radius;
    private transient int imageWidth;
    private transient int imageHeight;    
	
    public MedianFilter () {
	radius = 1;
    }
	
    public MedianFilter(int radius) {
        this.radius = radius;
    }        
	
    @Override
    public BufferedImage apply(BufferedImage image) {
		
        originalImage = image;
		
	imageWidth = originalImage.getWidth();
	imageHeight = originalImage.getHeight();
		
	filteredImage = new BufferedImage(imageWidth, imageHeight, originalImage.getType());
		
	int [] arrayOfPixels;
	int median;
	int alpha;
	int newColor;
		
	for (int i = 0; i < imageWidth; i++) {
            for (int j = 0; j < imageHeight; j++) {
				
		arrayOfPixels = getArrayOfPixels(i, j);
		median = findMedianPixelColor(arrayOfPixels);
		alpha = new Color(originalImage.getRGB(i, j)).getAlpha();
				
		newColor = ImageUtilities.argbToColor(alpha, median, median, median);
		filteredImage.setRGB(i, j, newColor);
            }
        }
		
	return filteredImage;
    }
	
    public int[] getArrayOfPixels (int x, int y) {
		
        int startX = x - radius;
	int goalX = x + radius;
	int startY = y - radius;
	int goalY = y + radius;
		
	if (startX < 0)
            startX = 0;
	if (goalX > originalImage.getWidth() - 1)
            goalX = originalImage.getWidth() - 1;
        if (startY < 0)
            startY = 0;
	if (goalY > originalImage.getHeight() - 1)
            goalY = originalImage.getHeight() - 1;
		
	int arraySize = (goalX - startX + 1)*(goalY - startY +1);
	int [] pixels = new int [arraySize];
		
	int position = 0;
	int color;
        for (int p = startX; p <= goalX; p++) {
            for (int q = startY; q <= goalY; q++) {
		color = new Color(originalImage.getRGB(p, q)).getRed();
		pixels[position] = color;
		position++;
            }
	}
		
	return pixels;
    }
	
    private int findMedianPixelColor (int [] arrayOfPixels) {
	Arrays.sort(arrayOfPixels);
	int middle = arrayOfPixels.length/2;
	return arrayOfPixels[middle];
    }

    /**
     * 
     * @param radius radius of the window. Current pixel is in center of this 
     * window 
     */
    
    public void setRadius(int radius) {
        this.radius = radius;
    }

    public int getRadius() {
        return radius;
    }
    
    

    @Override
    public String toString() {
        return "Median Filter";
    }

 
    
    
}