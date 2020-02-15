package org.neuroph.imgrec.filter;

import java.awt.image.BufferedImage;

/**
 * Interface for image filter 
 * @author Sanja
 * @param <T> Image class
 */
public interface ImageFilter<T> { 
    public BufferedImage apply(T image);
}
