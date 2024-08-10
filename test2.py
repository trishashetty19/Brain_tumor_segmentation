import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, confusion_matrix
from metrics import dice_loss, dice_coef
from train import load_dataset
import time

""" Global parameters """
H = 256
W = 256

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Load the model """
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(os.path.join("files", "model.keras"))

    """ Dataset """
    dataset_path = "../data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    """ Prediction and Evaluation """
    SCORE = []
    start_time = time.time()  # Start time recording

    for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
        """ Extracting the name """
        name = x.split("\\")[-1]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)  # [H, w, 3]
        image = cv2.resize(image, (W, H))       # [H, w, 3]
        x = image / 255.0                       # [H, w, 3]
        x = np.expand_dims(x, axis=0)           # [1, H, w, 3]

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        """ Prediction """
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = os.path.join("results", name)
        save_results(image, mask, y_pred, save_image_path)

        """ Flatten the array """
        mask_flat = mask.flatten()
        y_pred_flat = y_pred.flatten()

        """ Calculating the metrics values """
        accuracy = accuracy_score(mask_flat, y_pred_flat)
        mcc = matthews_corrcoef(mask_flat, y_pred_flat)
        roc_auc = roc_auc_score(mask_flat, y_pred_flat, average="macro")
        cohen_kappa = cohen_kappa_score(mask_flat, y_pred_flat)
        f1_value = f1_score(mask_flat, y_pred_flat)
        jac_value = jaccard_score(mask_flat, y_pred_flat)
        recall_value = recall_score(mask_flat, y_pred_flat)
        precision_value = precision_score(mask_flat, y_pred_flat)

        # Calculate specificity
        tn, fp, fn, tp = confusion_matrix(mask_flat, y_pred_flat, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

        # Calculate Dice coefficient
        dice = dice_coef(mask_flat, y_pred_flat)

        SCORE.append([name, accuracy, f1_value, jac_value, recall_value, precision_value, specificity, dice])

    """ Metrics values """
    end_time = time.time()  # End time recording
    elapsed_time = end_time - start_time

    score = np.mean(SCORE, axis=0)
    print(f"Mean Accuracy: {score[1]:0.5f}")
    print(f"Cohen's Kappa: {score[2]:0.5f}")
    print(f"MCC: {score[3]:0.5f}")
    print(f"F1: {score[4]:0.5f}")
    print(f"Jaccard: {score[5]:0.5f}")
    print(f"Recall: {score[6]:0.5f}")
    print(f"Precision: {score[7]:0.5f}")
    print(f"Specificity: {score[8]:0.5f}")
    print(f"Dice Coefficient: {score[9]:0.5f}")

    print(f"Test Time: {elapsed_time:.2f} seconds")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision", "Specificity", "Dice Coefficient"])
    df.to_csv("files/score.csv")
