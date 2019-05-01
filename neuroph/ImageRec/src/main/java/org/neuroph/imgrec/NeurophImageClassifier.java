package org.neuroph.imgrec;

import java.awt.image.BufferedImage;
import java.util.Map;
import javax.visrec.AbstractImageClassifier;
import org.neuroph.nnet.MultiLayerPerceptron;

public class NeurophImageClassifier extends AbstractImageClassifier<BufferedImage, MultiLayerPerceptron> {

    public NeurophImageClassifier(MultiLayerPerceptron model) {
        super(model);
    }

    @Override
    public Map<String, Float> classify(BufferedImage input) {
        // TODO: create vector from input image
        //getModel().setInput(inputVector);
        // return top5 mapiings for this class
        throw new UnsupportedOperationException("not implemented");
    }

}
