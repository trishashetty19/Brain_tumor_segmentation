import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Input
from tensorflow.keras.models import Model

def segnet_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_segnet5(input_shape):
    inputs = Input(input_shape)

    # Encoder
    e1 = segnet_block(inputs, 64)
    p1 = MaxPool2D((2, 2))(e1)

    e2 = segnet_block(p1, 128)
    p2 = MaxPool2D((2, 2))(e2)

    e3 = segnet_block(p2, 256)
    p3 = MaxPool2D((2, 2))(e3)

    e4 = segnet_block(p3, 512)
    p4 = MaxPool2D((2, 2))(e4)

    e5 = segnet_block(p4, 512)
    p5 = MaxPool2D((2, 2))(e5)

    # Decoder
    d5 = UpSampling2D((2, 2))(p5)
    d5 = segnet_block(d5, 512)
    
    d4 = UpSampling2D((2, 2))(d5)
    d4 = segnet_block(d4, 512)
    
    d3 = UpSampling2D((2, 2))(d4)
    d3 = segnet_block(d3, 256)
    
    d2 = UpSampling2D((2, 2))(d3)
    d2 = segnet_block(d2, 128)
    
    d1 = UpSampling2D((2, 2))(d2)
    d1 = segnet_block(d1, 64)
    
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d1)
    
    model = Model(inputs, outputs, name="SegNet5")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_segnet5(input_shape)
    model.summary()
