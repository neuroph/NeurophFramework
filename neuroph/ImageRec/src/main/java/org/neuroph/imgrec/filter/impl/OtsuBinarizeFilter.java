package org.neuroph.imgrec.filter.impl;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.Serializable;
import org.neuroph.imgrec.ImageUtilities;
import org.neuroph.imgrec.filter.ImageFilter;

/**
 * Otsu binarize filter serves to dynamically determine the threshold based on 
 * the whole image and for later binarization on black (0) and white (255) pixels. 
 * In determining threshold a image histogram is created in way that the value of 
 * each pixel of image affects on the histogram appearance. Then, depending upon 
 * the look of the histogram threshold counts and based on that, the real image 
 * which is binarized is made.The image before this filter MUST be GRAYSCALE and 
 * at the end image will contain only two colors - black and white. 
 *
 * threshold - could be trainable
 * 
 *  Reference to: http://zerocool.is-a-geek.net/?p=376
 *  http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
 * 
 * @author Mihailo Stupar
 */
public class OtsuBinarizeFilter implements ImageFilter<BufferedImage>, Serializable {

    private transient BufferedImage originalImage;
    private transient BufferedImage filteredImage;
        
    private static final int BLACK_PIXEL = 0;
    private static final int WHITE_PIXEL = 255;    
    
    @Override	
    public BufferedImage apply(BufferedImage image) {		
         this.originalImage = image;		
	int width = originalImage.getWidth();
	int height = originalImage.getHeight();		
	filteredImage = new BufferedImage(width, height, originalImage.getType());
		
	int [] histogram = imageHistogram(originalImage);		
	int totalNumberOfpixels = width * height;		
	int threshold = calculateThreshold(histogram, totalNumberOfpixels); // ovo moze biti adaptivni parametar - adaptivna konvoluciona binarizacija			
	int alpha, gray, newColor;
		
	for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                 final Color col = new Color(originalImage.getRGB(x, y));
		gray = col.getRed();
		alpha = col.getAlpha();
				
		if (gray > threshold)
                    newColor = WHITE_PIXEL;
		else
                    newColor = BLACK_PIXEL;
				
		newColor = ImageUtilities.argbToColor(alpha, newColor, newColor, newColor);
		filteredImage.setRGB(x, y, newColor);
            }
	}
		
	return filteredImage;
    }
	
    /**
     * Creates image histogram for the given image.
     * 
     * @param image
     * @return 
     */
    private int[] imageHistogram(BufferedImage image) {
        int[] histogram = new int[256];

        for (int x = 0; x < image.getWidth(); x++) {
            for (int y = 0; y < image.getHeight(); y++) {
                // ovde bi bilo bolje da ne instanciram ovoliko objekata... nego nek atransformacija: (getRGB() >> 16) & 0xFF to se radi u getRed()
                int gray = (image.getRGB(x, y) >> 16) & 0xFF; // ovo je getRed() samo crveni kanal jer se pretpostavlja da je prethodno prosla kroz grayscale filter pa su sva tri kaala ista
                histogram[gray]++; // za svaku nijansu sive povecaj odgovarajucu poziciju za 1
            }
        }
        return histogram;
    }
	
    private int calculateThreshold(int[] histogram, int total) {
        float sum = 0;
        for (int i = 0; i < 256; i++) {
            sum += i * histogram[i];
        }

        float sumB = 0;
        int wB = 0;
        int wF = 0;

        float varMax = 0;
        int threshold = 0;

        for (int i = 0; i < 256; i++) {
            wB += histogram[i];
            if (wB == 0) {
                continue;
            }
            wF = total - wB;

            if (wF == 0) {
                break;
            }

            sumB += (float) (i * histogram[i]);
            float mB = sumB / wB;
            float mF = (sum - sumB) / wF;

            float varBetween = (float) wB * (float) wF * (mB - mF) * (mB - mF);

            if (varBetween > varMax) {
                varMax = varBetween;
                threshold = i;
            }
        }
        return threshold;
    }
    
    @Override
    public String toString() {
        return "Otsu Binarize Filter";
    }

 
}