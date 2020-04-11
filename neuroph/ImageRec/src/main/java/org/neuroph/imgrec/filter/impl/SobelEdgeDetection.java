package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 * Edge detection using sobel filter
 * 
 * https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm 
 * @author Mihailo Stupar
 */
public class SobelEdgeDetection implements ImageFilter<BufferedImage>, Serializable{
    
    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;
    
    private double [][] horizSobel;
    private double [][] vertSobel;    
    private double threshold;    
    public static final int WHITE = 255;
    public static final int BLACK = 0;       

    public SobelEdgeDetection() {
        initSobelFilters();   
        threshold = 0.1;
    }
    
    @Override
    public BufferedImage apply(BufferedImage image) {
        
        originalImage = image;        
        int width = image.getWidth();
        int height = image.getHeight();        
        filteredImage = new BufferedImage(width, height, image.getType());        
        
        double [][] gradX = new double[width][height];
        double [][] gradY = new double[width][height];
        double [][] grad = new double[width][height];
        
        double maxGrad = 0;
        
        for (int x = 1; x < width-1; x++) {
            for (int y = 1; y < height-1; y++) {                
                gradX[x][y] = applyFilter(x, y, horizSobel);
                gradY[x][y] = applyFilter(x, y, vertSobel);                
                grad[x][y] = Math.abs(gradX[x][y]) + Math.abs(gradY[x][y]); // aproksimacija intenziteta gradijenta               
                if (grad[x][y] > maxGrad)  maxGrad = grad[x][y];
            }            
        }
                
        threshold = threshold * maxGrad;
        int newPixelColor;
        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {                               
                final int alpha = new Color(originalImage.getRGB(x, y)).getAlpha();
                
                if (grad[x][y] > threshold) newPixelColor = BLACK;  // ako je promena/gradijent veci onda crni pixel
                    else newPixelColor = WHITE; // ako nije ond abeli pixel
               
                final int rgb = ImageUtilities.argbToColor(alpha, newPixelColor, newPixelColor, newPixelColor);
                filteredImage.setRGB(x, y, rgb);                
            }
        }
      
        return filteredImage;
    }
    
    private void initSobelFilters () {
                
        horizSobel = new double[3][3];
        horizSobel [0][0] = -1;  horizSobel [0][1] = -2;  horizSobel [0][2] = -1;
        horizSobel [1][0] = 0;      horizSobel [1][1] = 0;     horizSobel [1][2] = 0;
        horizSobel [2][0] = 1;   horizSobel [2][1] = 2;   horizSobel [2][2] = 1;
        
        vertSobel = new double[3][3];
        vertSobel [0][0] = -1;  vertSobel [0][1] = 0;  vertSobel [0][2] = 1;
        vertSobel [1][0] = -2;   vertSobel [1][1] = 0;  vertSobel [1][2] = 2;
        vertSobel [2][0] = -1;  vertSobel [2][1] = 0;  vertSobel [2][2] = 1;        
        
    }
    
    // applies filter at specified position - calculateGradient
    protected double applyFilter(int xCenter, int yCenter, double[][] sobelFilter) {           
    	double filterSum = 0;
        
        int fx = 0;
        for (int x = xCenter-1; x <= xCenter+1; x++) {            
            int fy = 0;
            for (int y = yCenter-1; y <= yCenter+1; y++) {                
                final double pixelGray = new Color(originalImage.getRGB(x, y)).getRed();                
                filterSum = filterSum + pixelGray*sobelFilter[fx][fy];
                fy++;
            }
            fx++;
        }
        
        return filterSum;
    } 

    public void setTreshold(double treshold) {
        this.threshold = treshold;
    }

    public double getThreshold() {
        return threshold;
    }       

    @Override
    public String toString() {
        return "Sobel method";
    }
    
    
}