import os
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
import signal
import sys

from unet import build_unet            # unet
from resnet import build_resnet_unet   # resnet
from segunet import build_segunet      # segunet
from usegnet import build_usegnet      # usegnet
from segnet3 import build_segnet3      # segnet3
from segnet5 import build_segnet5      # segnet5
from res_segnet import build_res_segnet # res-segnet
from unet_resnet import build_hybrid_unet_resnet # unet_resnet 


from metrics import dice_loss, dice_coef

""" Global parameters """
H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    split_size = split  # Ensure split_size remains a float

    train_x, test_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(masks, test_size=split_size, random_state=42)
	
    # split_size = split / (1 - split)  # Adjust split_size for further splitting train set into train and test

    temp_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    temp_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (h, w)
    x = cv2.resize(x, (W, H))   ## (h, w)
    x = x / 255.0               ## (h, w)
    x = x.astype(np.float32)    ## (h, w)
    x = np.expand_dims(x, axis=-1)## (h, w, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

class GracefulExitCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GracefulExitCallback, self).__init__()
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\nInterrupting at the end of this epoch.")
        self.interrupted = True

    def on_epoch_end(self, epoch, logs=None):
        if self.interrupted:
            self.model.stop_training = True
            print("\nTraining interrupted gracefully at the end of epoch.")
            sys.exit(0)

class PrintValMetrics(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy')
        val_dice_coef = logs.get('val_dice_coef')
        print(f"Epoch {epoch+1}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation Dice Coefficient: {val_dice_coef}")

if __name__ == "__main__":
    """ Seeding """
    print(tf.config.list_physical_devices('GPU'))

    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-4  # Set desired learning rate
    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "log.csv")

    """ Dataset """
    dataset_path = "../data"  # Ensure this path is correct
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    """ Choose and build model """
    model =  build_unet((H, W, 3))  # Replacing with desired model function
    model.compile(loss=dice_loss, optimizer=Adam(learning_rate), metrics=['accuracy', dice_coef])

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    """ Training """
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=[
            ModelCheckpoint(model_path, verbose=1, save_best_only=True, save_weights_only=False),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
            CSVLogger(csv_path),
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
            # GracefulExitCallback(),
            PrintValMetrics()
        ]
    )

