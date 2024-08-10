import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.applications import VGG16 # type: ignore

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

def build_segunet(input_shape):
    inputs = Input(input_shape)

    vgg = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encoder: Extract skip connections
    s1 = vgg.get_layer("block1_conv2").output
    s2 = vgg.get_layer("block2_conv2").output
    s3 = vgg.get_layer("block3_conv3").output
    s4 = vgg.get_layer("block4_conv3").output
    b1 = vgg.get_layer("block5_conv3").output

    # Decoder: Upsampling and concatenation
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="SegUNet")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_segunet(input_shape)
    model.summary()
