package org.neuroph.imgrec.filter.impl;


import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 *
 * @author Mihailo Stupar
 */
public class GaussianNoise implements ImageFilter<BufferedImage>,Serializable{

    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;
    
    private double mean;
    private double sigma;

    public GaussianNoise() {
        mean = 0;
        sigma = 30;
    }
    
    
    
    @Override
    public BufferedImage apply(BufferedImage image) {
        double variance = sigma*sigma;
        originalImage = image;
        
        int width = originalImage.getWidth();
        int height = originalImage.getHeight();
        filteredImage = new BufferedImage(width, height, originalImage.getType());
        
        double a = 0.0;
        double b = 0.0;
        
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {              
                while (a == 0.0) // sta je ovo ovde jel tu nedosteje nesto ili ga treba izbaciti??? kO DA NEMA nikakvu svrhu samo da a mora biti razlicito od nule
                    a = Math.random();
                b = Math.random();
                
                double x = Math.sqrt(-2*Math.log(a)) * Math.cos(2*Math.PI*b);
                double noise = mean + Math.sqrt(variance) * x;
                
                int gray = new Color(originalImage.getRGB(i, j)).getRed();
                int alpha = new Color(originalImage.getRGB(i, j)).getAlpha();
                
                double color = gray + noise;
                if (color > 255) color = 255;
                if (color < 0) color = 0;
                
                int newColor = (int) Math.round(color);                
                int newRGB = ImageUtilities.argbToColor(alpha, newColor, newColor, newColor);
                
                filteredImage.setRGB(i, j, newRGB);                                
            }
        }
                
        return filteredImage;
    }

    public void setMean(double mean) {
        this.mean = mean;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    @Override
    public String toString() {
        return "Gaussian noise";
    }

    
    
    
    
    
}
