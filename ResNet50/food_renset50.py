import cv2
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import relu, softmax
from sklearn.metrics import confusion_matrix
import json
from google.colab.patches import cv2_imshow


class FoodResNet50:
    def __init__(self, train_dir, val_dir, test_dir):
        try:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                shear_range=0.2,
                zoom_range=0.2,
                rotation_range=30,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
            )

            val_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
            )

            test_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
            )

            # Loading the data
            self.train_data = train_datagen.flow_from_directory(
                train_dir,
                target_size=(224, 224),
                batch_size=20,
                class_mode='categorical',
                color_mode='rgb'
            )

            self.val_data = val_datagen.flow_from_directory(
                val_dir,
                target_size=(224, 224),
                batch_size=10,
                class_mode='categorical',
                color_mode='rgb'
            )

            self.test_data = test_datagen.flow_from_directory(
                test_dir,
                target_size=(224, 224),
                batch_size=10,
                class_mode='categorical',
                color_mode='rgb'
            )

            # Class labels
            self.labels = list(self.train_data.class_indices.keys())
            print(f'Classes: {self.labels}')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def build_resnet50(self):
        try:
            # ResNet50 base model
            base_model = ResNet50(include_top = False, input_shape = (224, 224, 3), weights = 'imagenet')

            for i in base_model.layers:
                i.trainable = False

            self.model = Sequential([
                base_model,
                Flatten(),
                Dense(256, activation = relu),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation = relu),
                BatchNormalization(),
                Dense(len(self.labels), activation = softmax)
            ])

            self.model.summary()

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def train_model(self, epochs):
        try:
            # Training the model
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model.fit(
                self.train_data,
                validation_data = self.val_data,
                epochs = epochs
            )

            print('Training completed.')

            self.model.save('resnet50_model.h5')

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def evaluate_model(self, filename):
        try:
            test_loss, test_accuracy =self.model.evaluate(self.test_data)

            print(f'Test Loss: {test_loss:.4f}')
            print(f'Test Accuracy: {test_accuracy*100:.4f}')

            y_pred_prob = self.model.predict(self.test_data)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = self.test_data.classes

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)

            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            TN = np.sum(cm) - (TP+FP+FN)

            metrics_dict = {
                "per_class": {},
                "overall": {},
                "metrics": {}
            }

            for i in range(len(TP)):
                metrics_dict['per_class'][f'class_{i}'] = {
                    'TP': int(TP[i]),
                    'FP': int(FP[i]),
                    'FN': int(FN[i]),
                    'TN': int(TN[i])
                }

                overall_TP = np.sum(TP)
                overall_FP = np.sum(FP)
                overall_FN = np.sum(FN)
                overall_TN = np.sum(TN)

                metrics_dict['overall'] = {
                    'TP': int(overall_TP),
                    'FP': int(overall_FP),
                    'FN': int(overall_FN),
                    'TN': int(overall_TN)
                }

                # Calculating metrics
                acc = (overall_TP + overall_TN) / (overall_TP + overall_TN + overall_FP + overall_FN)
                precision = overall_TP / (overall_TP + overall_FP)
                recall = overall_TP / (overall_TP + overall_FN)
                f_score = 2 * (precision * recall) / (precision + recall)

                metrics_dict['metrics'] = {
                    'Accuracy': acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f_score
                }

                with open(filename, 'w') as f:
                    json.dump(metrics_dict, f, indent=4)

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')

    def checking(self, file_path):
        try:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (224, 224))
            img_rescaled = resized_img / 255.0
            final_image = np.expand_dims(img_rescaled, axis=0)
            final_image = np.expand_dims(final_image, axis=-1)

            output = self.model.predict(final_image)
            print(f'Predicted Class: {self.labels[np.argmax(output)]}')

            cv2_imshow(resized_img)  # Only works in google colab
            # cv2.imshow('Test Image', resized_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')


if __name__ == '__main__':
    try:
        train_path = r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Training'
        val_path = r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Validation'
        test_path = r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification Dataset New\Testing'
        food_resnet = FoodResNet50(train_path, val_path, test_path)
        food_resnet.build_resnet50()
        food_resnet.train_model(epochs=30)
        food_resnet.evaluate_model(filename='resnet50_metrics.json')
        food_resnet.checking(file_path=r'C:\Users\Admin\Desktop\Internship\Task-1\Food Classification dataset\burger\001.jpg')

    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        print(f'Error Type: {er_type}\nError Msg: {er_msg}\nError Line: {er_line.tb_lineno}')