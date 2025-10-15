import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Add, Input
from tensorflow.keras.models import Model
import mlflow
import os
from getpass import getpass
from keras.callbacks import ModelCheckpoint, EarlyStopping

############### Inicializamos conectividad con el repositorio "Experimentos" en DagsHub ###########
REPO_NAME= "Experimentos"
REPO_OWNER= "carlosfcfm"  #Escribir nombre de repositorio
USER_NAME = "carlosfcfm" #Escribir su usuario
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token or password: ')
mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
###################################################################################################

df = pd.read_csv(r'C:\Users\LamdaZero\Desktop\Pruebas 2\global_house_purchase_dataset.csv')
df = df.drop('property_id', axis=1)

# Codificamos variables categóricas con one-hot encoding simple usando pandas
categorical_cols = ['country', 'city', 'property_type', 'furnishing_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=int)

# Separamos características (X) y etiqueta (y)
X = df_encoded.drop('decision', axis=1).values  # Convertimos a un array de numpy
y = df_encoded['decision'].values

# Normalizamos las características numéricas (escalado simple entre 0 y 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Dividimos primero en 70% entrenamiento y 30% un conjuno temporal (prueba + validación)
X_train, X_pv, y_train, y_pv = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Luego dividimos el 30% del conjunto temporal en 20% prueba y 10% validación
# 20/30 = 2/3 (prueba), 10/30 = 1/3 (validación)
X_test, X_val, y_test, y_val = train_test_split(X_pv, y_pv, test_size=1/3, random_state=42)
# Obtenemos el número de características
num_features = X_train.shape[1]
print("El número de características es: ", num_features)
# Hacemos el reshape para CNN 1D con forma: (muestras, características, 1 canal)
X_train = X_train.reshape((X_train.shape[0], num_features, 1))
X_test = X_test.reshape((X_test.shape[0], num_features, 1))
# X_val = X_val.reshape((X_val.shape[0], num_features, 1))

# Parámetros fijos (basados en el mejor trial de Optuna)
epochs = 40
batch_size = 64
filters_1 = 128
filters_2 = 192
filters_3 = 384
kernel_size = 3
dense_units = 512
l1_value = 2.1179368863512548e-05  
learning_rate = 4.808589316209705e-05
optimizer_name = "adam"

## Modelo CONV1D
inputs = keras.Input(shape=(num_features, 1))

# Primer bloque residual
x = layers.Conv1D(filters=filters_1, kernel_size=kernel_size, padding='same')(inputs)
x = layers.Activation('relu')(x)
x = layers.Conv1D(filters=filters_1, kernel_size=kernel_size, padding='same')(x)
x = layers.Activation('relu')(x)
residual = x
x = layers.Add()([x, residual])
x = layers.MaxPooling1D(pool_size=2)(x)

# Segundo bloque residual
x = layers.Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
x = layers.Conv1D(filters=filters_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
residual = x
x = layers.Add()([x, residual])
x = layers.MaxPooling1D(pool_size=2)(x)

# Tercer bloque residual
x = layers.Conv1D(filters=filters_3, kernel_size=kernel_size, activation='relu', padding='same')(x)
x = layers.Conv1D(filters=filters_3, kernel_size=kernel_size, activation='relu', padding='same')(x)
residual = x
x = layers.Add()([x, residual])
x = layers.MaxPooling1D(pool_size=2)(x)

# Capas finales
x = layers.Flatten()(x)
x = layers.Dense(dense_units, activation='relu', kernel_regularizer=regularizers.l1(l1_value))(x)
outputs = layers.Dense(1, activation='sigmoid')(x)


# Definimos el modelo
model = Model(inputs=inputs, outputs=outputs)

# Configuramos el optimizador
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

mlflow.set_experiment("Tarea5_Modelos_finales")
with mlflow.start_run(run_name="Mejor modelo L1"):
    mlflow.tensorflow.autolog(log_models=False)

    # Loggeamos hiperparámetros en Mlflow
    mlflow.log_params({
        "filters_1": filters_1,
        "filters_2": filters_2,
        "filters_3": filters_3,
        "kernel_size": kernel_size,
        "dense_units": dense_units,
        "l1_value ": l1_value ,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "optimizer": optimizer_name
    })

    # Callbacks
    filepath = "best_modelconv_l1.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=5, verbose=1)
    callbacks = [earlystop, checkpoint]

    # Entrenar el modelo
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)

    model.save(filepath)
    mlflow.log_artifact(filepath, artifact_path="models")
    mlflow.end_run()
    


