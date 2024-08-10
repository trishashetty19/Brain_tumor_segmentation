import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_usegnet(input_shape):
    inputs = Input(input_shape)

    # Encoder (SegNet part)
    s1 = Conv2D(64, 3, activation="relu", padding="same")(inputs)
    s1 = BatchNormalization()(s1)
    s1 = Conv2D(64, 3, activation="relu", padding="same")(s1)
    s1 = BatchNormalization()(s1)
    p1 = MaxPool2D((2, 2))(s1)

    s2 = Conv2D(128, 3, activation="relu", padding="same")(p1)
    s2 = BatchNormalization()(s2)
    s2 = Conv2D(128, 3, activation="relu", padding="same")(s2)
    s2 = BatchNormalization()(s2)
    p2 = MaxPool2D((2, 2))(s2)

    s3 = Conv2D(256, 3, activation="relu", padding="same")(p2)
    s3 = BatchNormalization()(s3)
    s3 = Conv2D(256, 3, activation="relu", padding="same")(s3)
    s3 = BatchNormalization()(s3)
    p3 = MaxPool2D((2, 2))(s3)

    s4 = Conv2D(512, 3, activation="relu", padding="same")(p3)
    s4 = BatchNormalization()(s4)
    s4 = Conv2D(512, 3, activation="relu", padding="same")(s4)
    s4 = BatchNormalization()(s4)
    p4 = MaxPool2D((2, 2))(s4)

    b1 = Conv2D(1024, 3, activation="relu", padding="same")(p4)
    b1 = BatchNormalization()(b1)
    b1 = Conv2D(1024, 3, activation="relu", padding="same")(b1)
    b1 = BatchNormalization()(b1)

    # Decoder (UNet part)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-SegNet")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_usegnet(input_shape)
    model.summary()
