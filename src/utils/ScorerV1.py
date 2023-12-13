import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import datasets
from sklearn import metrics



class ScorerV1:
    def __init__(self,X_train,y_train,X_test,y_test) -> None:
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = X_test
        self.y_test = y_test
        
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.X_train.shape[1], activation='relu'),  
            tf.keras.layers.Dropout(0.5),  # Agregar Dropout para regularización
            tf.keras.layers.Dense(32, activation='relu'),  # Capa oculta adicional
            tf.keras.layers.Dense(8, activation=None),  # Capa oculta adicional
            tf.keras.layers.Dense(1)  
        ])
        
        self.net.compile(optimizer='adam', loss='mean_squared_error')
        self.train()
        self.y_pred = self.net.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        print(f'Error Cuadrático Medio en conjunto de prueba: {self.mse:.4f}')
        
    def train(self):
        for epoch in range(30):
            # Entrenar el modelo en una época
            history = self.net.fit(self.X_train, self.y_train, epochs=1, batch_size=16, verbose=0)

            # Calcular la pérdida en el conjunto de entrenamiento y conjunto de prueba
            train_loss = self.net.evaluate(self.X_train, self.y_train, verbose=0)
            test_loss = self.net.evaluate(self.X_test,self.y_test, verbose=0)

            # Realizar predicciones en el conjunto de prueba
            y_pred = self.net.predict(self.X_test)

            # Calcular el error absoluto medio en el conjunto de prueba
            mae = mean_absolute_error(self.y_test, y_pred)

            # Imprimir resultados después de cada época
            print(f'Época {epoch + 1}/{30} - Pérdida en entrenamiento: {train_loss:.4f}, Pérdida en prueba: {test_loss:.4f}, MAE: {mae:.4f}')
