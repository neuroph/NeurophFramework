/**
 * Copyright 2010 Neuroph Project http://neuroph.sourceforge.net
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package org.neuroph.util;

import java.util.Iterator;
import org.neuroph.core.Neuron;
import org.neuroph.core.input.InputFunction;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.Linear;
import org.neuroph.core.transfer.TransferFunction;

/**
 * Represents properties of a neuron.
 *
 * @author Zoran Sevarac <sevarac@gmail.com>
 */
public class NeuronProperties extends NeurophProperties {

    private static final long serialVersionUID = 3L;
    
    public final static String INPUT_FUNCTION = "inputFunction";
    public final static String TRANSFER_FUNCTION = "transferFunction";
    public final static String NEURON_TYPE = "neuronType";
    public final static String USE_BIAS = "useBias";

    public NeuronProperties() {
        initKeys();
        this.setProperty(INPUT_FUNCTION, WeightedSum.class);
        this.setProperty(TRANSFER_FUNCTION, Linear.class);
        this.setProperty(NEURON_TYPE, Neuron.class);
    }

    public NeuronProperties(Class<? extends Neuron> neuronClass) {
        initKeys();
        this.setProperty(INPUT_FUNCTION, WeightedSum.class);
        this.setProperty(TRANSFER_FUNCTION, Linear.class);
        this.setProperty(NEURON_TYPE, neuronClass);
    }

    public NeuronProperties(Class<? extends Neuron> neuronClass, Class<? extends TransferFunction> transferFunctionClass) {
        initKeys();
        this.setProperty(INPUT_FUNCTION, WeightedSum.class);
        this.setProperty(TRANSFER_FUNCTION, transferFunctionClass);
        this.setProperty(NEURON_TYPE, neuronClass);
    }

    public NeuronProperties(Class<? extends Neuron> neuronClass,
            Class<? extends InputFunction> inputFunctionClass,
            Class<? extends TransferFunction> transferFunctionClass) {
        initKeys();
        this.setProperty(INPUT_FUNCTION, inputFunctionClass);
        this.setProperty(TRANSFER_FUNCTION, transferFunctionClass);
        this.setProperty(NEURON_TYPE, neuronClass);
    }

    public NeuronProperties(Class<? extends Neuron> neuronClass, TransferFunctionType transferFunctionType) {
        initKeys();
        this.setProperty(INPUT_FUNCTION, WeightedSum.class);
        this.setProperty(TRANSFER_FUNCTION, transferFunctionType.getTypeClass());
        this.setProperty(NEURON_TYPE, neuronClass);
    }

    public NeuronProperties(TransferFunctionType transferFunctionType, boolean useBias) {
        initKeys();
        this.setProperty(INPUT_FUNCTION, WeightedSum.class);
        this.setProperty(TRANSFER_FUNCTION, transferFunctionType.getTypeClass());
        this.setProperty(USE_BIAS, useBias); // ovo bi trebalo da je defaultno podesavanje uvek na tru, ako nece moze da se stavi na 0
        this.setProperty(NEURON_TYPE, Neuron.class);
    }

    // uraditi override za setProperty tako da za enum type uzima odgovarajuce klase
    private void initKeys() {
        createKeys(INPUT_FUNCTION, TRANSFER_FUNCTION, NEURON_TYPE, USE_BIAS); // use bias prebaciti u NeuralNetworkProperties
    }

    public Class getInputFunction() {
        Object val = this.get(INPUT_FUNCTION);
        if (!val.equals("")) {
            return (Class) val;
        }
        return null;
    }

    public Class getTransferFunction() {
        return (Class) this.get(TRANSFER_FUNCTION);
    }

    public Class getNeuronType() {
        return (Class) this.get(NEURON_TYPE);
    }

    public NeurophProperties getTransferFunctionProperties() {
        NeurophProperties tfProperties = new NeurophProperties();
        //Enumeration<?> en = this.keys(); 
        Iterator iterator =  this.keySet().iterator();
        while (iterator.hasNext()) {
            String name = iterator.next().toString();
            if (name.contains(TRANSFER_FUNCTION)) {
                tfProperties.setProperty(name, this.get(name));
            }
        }
        return tfProperties;
    }

    @Override
    public final void setProperty(String key, Object value) {
        if (value instanceof TransferFunctionType) {
            value = ((TransferFunctionType) value).getTypeClass();
        }
        //      if (value instanceof InputFunctionType) value = ((InputFunctionType)value).getTypeClass();

        this.put(key, value);
    }
}