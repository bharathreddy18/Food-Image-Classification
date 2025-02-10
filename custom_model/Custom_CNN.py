import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2
from google.colab.patches import cv2_imshow
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json


class CNN:
    def __init__(self, train_dir, test_dir, val_dir):
        try:
            temp_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

            temp_data = temp_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=20,
                class_mode='categorical',
                color_mode='grayscale'
            )

            x_sample, _ = next(temp_data)

            # Data Augmentation
            train_datagen = ImageDataGenerator(
                rescale = 1.0/255.0,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                rotation_range = 30,
                width_shift_range = 0.2,
                height_shift_range = 0.2,
                featurewise_center=True,
                featurewise_std_normalization=True
            )

            train_datagen.fit(x_sample)

            val_test_datagen = ImageDataGenerator(
                rescale = 1.0/255.0,
                featurewise_center=True,
                featurewise_std_normalization=True
            )

            val_test_datagen.mean = train_datagen.mean
            val_test_datagen.std = train_datagen.std


            # Load training, validation and testing datasets.
            self.train_data = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224,224),
                batch_size=20,
                class_mode='categorical',
                color_mode='grayscale'
            )

            self.val_data = val_test_datagen.flow_from_directory(
                val_dir,
                target_size=(224,224),
                batch_size=10,
                class_mode='categorical',
                color_mode='grayscale'
            )

            self.test_data = val_test_datagen.flow_from_directory(
                test_dir,
                target_size=(224,224),
                batch_size=10,
                class_mode='categorical',
                color_mode='grayscale'
            )


            # Class labels
            self.labels = list(self.train_data.class_indices.keys())
            print(f'Classes: {self.labels}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def architecture(self):
        try:
            # Initializing the model
            self.model = Sequential()

            # Convolutional and Pooling layers
            self.model.add(Conv2D(256, kernel_size=(3,3), activation=relu, input_shape=(224,224,1), padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(128, kernel_size=(3,3), activation=relu, padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Dropout(0.25))

            self.model.add(Conv2D(64, kernel_size=(3,3), activation=relu, padding='same'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Dropout(0.25))

            # Convert 3D features to 1D
            self.model.add(Flatten())

            # Fully connected dense layers
            self.model.add(Dense(64, activation=relu))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation=relu))
            self.model.add(BatchNormalization())
            self.model.add(Dense(len(self.labels), activation=softmax))

            self.model.summary()

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def metrics(self):
        try:
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model.fit(
                self.train_data,
                validation_data = self.val_data,
                epochs = 50
            )

            self.model.save('custom_cnn.h5')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def evaluate(self, filename):
        try:
            self.y_true = self.test_data.classes                    # Actual class labels
            self.y_pred_prob = self.model.predict(self.test_data)   # Predicted probabilities
            self.y_pred = np.argmax(self.y_pred_prob, axis=1)       # predicted class labels

            # confusion matrix
            cm = confusion_matrix(self.y_true, self.y_pred)

            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            TN = np.sum(cm) - (TP+FP+FN)

            # store data in dictionary
            metrics_dist = {'per_class': {}, 'overall': {}, 'metrics': {}}

            for i in range(len(TP)):
                metrics_dist['per_class'][f'class_{i}'] = {
                    'TP': int(TP[i]),
                    'TN': int(TN[i]),
                    'FP': int(FP[i]),
                    'FN': int(FN[i])
                }

            # Overall TP, TN, FP, FN
            overall_TP = np.sum(TP)
            overall_TN = np.sum(TN)
            overall_FP = np.sum(FP)
            overall_FN = np.sum(FN)

            metrics_dist['overall'] = {
                'TP': int(overall_TP),
                'TN': int(overall_TN),
                'FP': int(overall_FP),
                'FN': int(overall_FN)
            }

            accuracy = (overall_TP + overall_TN) / (overall_TP + overall_TN + overall_FP + overall_FN)
            precision = overall_TP / (overall_TP + overall_FP)
            recall = overall_TP / (overall_TP + overall_FN)
            f_score = 2 * (precision * recall) / (precision + recall)

            metrics_dist['metrics'] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f_score
            }

            with open(filename, 'w') as json_file:
                json.dump(metrics_dist, json_file, indent=4)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def checking(self, file_path):
        try:
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img, (224,224))
            img_rescaled = resized_img / 255.0
            final_image = np.expand_dims(img_rescaled, axis=0)
            final_image = np.expand_dims(final_image, axis=-1)

            output = self.model.predict(final_image)
            print(f'Predicted Class: {self.labels[np.argmax(output)]}')

            cv2_imshow(resized_img)                         # Only works in google colab
            # cv2.imshow('Test Image', resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

if __name__ == "__main__":
    try:
        train_dir = r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Training'
        test_dir = r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Testing'
        val_dir = r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Validation'
        food_cnn = CNN(train_dir, test_dir, val_dir)
        food_cnn.architecture()
        food_cnn.metrics()
        food_cnn.evaluate(filename='cnn_metrics.json')
        food_cnn.checking(file_path=r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Validation\Donut\Donut (5).jpeg')

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')