import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

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

def build_hybrid_unet_efficientnet(input_shape):
    inputs = Input(input_shape)

    efficientnet = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encoder: Extract skip connections from EfficientNetB0
    s1 = efficientnet.get_layer("block2a_expand_activation").output   # 24 filters
    s2 = efficientnet.get_layer("block3a_expand_activation").output   # 32 filters
    s3 = efficientnet.get_layer("block4a_expand_activation").output   # 48 filters
    s4 = efficientnet.get_layer("block6a_expand_activation").output   # 136 filters
    b1 = efficientnet.get_layer("top_activation").output              # 1280 filters

    # Decoder: Upsampling and concatenation
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="Hybrid_UNET_EfficientNetB0")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_hybrid_unet_efficientnet(input_shape)
    model.summary()
