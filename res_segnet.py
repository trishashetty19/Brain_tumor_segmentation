import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Input, Add
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.models import Model

def residual_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(num_filters, 1, padding="same")(inputs)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x

def segnet_block(inputs, num_filters):
    x = residual_block(inputs, num_filters)
    x = residual_block(x, num_filters)
    return x

def build_res_segnet(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    base_model = ResNet18(include_top=False, weights='imagenet', input_tensor=inputs)
    skip1 = base_model.get_layer('conv1_relu').output
    skip2 = base_model.get_layer('conv2_block1_out').output
    skip3 = base_model.get_layer('conv3_block1_out').output
    skip4 = base_model.get_layer('conv4_block1_out').output
    skip5 = base_model.get_layer('conv5_block1_out').output

    # Decoder
    d5 = UpSampling2D((2, 2))(skip5)
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

    model = Model(inputs, outputs, name="Res-SegNet")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = build_res_segnet(input_shape)
    model.summary()
