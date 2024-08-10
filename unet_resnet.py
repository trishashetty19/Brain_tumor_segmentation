import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50

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

def build_hybrid_unet_resnet(input_shape):
    inputs = Input(input_shape)

    resnet = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    # Encoder: Extract skip connections from ResNet
    s1 = resnet.get_layer("conv1_relu").output            # 64 filters
    s2 = resnet.get_layer("conv2_block3_out").output      # 256 filters
    s3 = resnet.get_layer("conv3_block4_out").output      # 512 filters
    s4 = resnet.get_layer("conv4_block6_out").output      # 1024 filters
    b1 = resnet.get_layer("conv5_block3_out").output      # 2048 filters

    # Decoder: Upsampling and concatenation
    d1 = decoder_block(b1, s4, 1024)
    d2 = decoder_block(d1, s3, 512)
    d3 = decoder_block(d2, s2, 256)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="Hybrid_UNET_ResNet")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_hybrid_unet_resnet(input_shape)
    model.summary()
