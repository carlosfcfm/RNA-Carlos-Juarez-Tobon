import optuna
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

# Inicializamos conectividad con el repositorio "Experimentos" en DagsHub
REPO_NAME = "Experimentos"
REPO_OWNER = "carlosfcfm"  # Escribir nombre de repositorio
USER_NAME = "carlosfcfm"  # Escribir su usuario
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token or password: ')
mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
################################################################################

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
X_val = X_val.reshape((X_val.shape[0], num_features, 1))

def create_model(trial):
    inputs = keras.Input(shape=(num_features, 1))

    # Buscamos con optuna el kernel_size
    kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)

    # Buscamos con optuna los filtros para cada bloque
    filtros_1 = trial.suggest_int("filtros_1", 32, 128, step=32)
    filtros_2 = trial.suggest_int("filtros_2", 64, 256, step=64)
    filtros_3 = trial.suggest_int("filtros_3", 128, 512, step=128)

    # Primer bloque residual
    x = layers.Conv1D(filters=filtros_1, kernel_size=kernel_size, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(filters=filtros_1, kernel_size=kernel_size, padding='same')(x)
    x = layers.Activation('relu')(x)    
    residual = x  
    x = layers.Add()([x, residual])  # Conexión residual
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Segundo bloque residual
    x = layers.Conv1D(filters=filtros_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = layers.Conv1D(filters=filtros_2, kernel_size=kernel_size, activation='relu', padding='same')(x)
    residual = x
    x = layers.Add()([x, residual])  # Conexión residual
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Tercer bloque residual
    x = layers.Conv1D(filters=filtros_3, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = layers.Conv1D(filters=filtros_3, kernel_size=kernel_size, activation='relu', padding='same')(x)
    residual = x
    x = layers.Add()([x, residual])  # Conexión residual
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Capa final
    x = layers.Flatten()(x)

    # Buscamos con optuna las unidades en Dense
    dense_units = trial.suggest_int("dense_units", 128, 512, step=64)
    l1_value = trial.suggest_float("l1_value", 1e-5, 1e-2, log=True)
    x = layers.Dense(dense_units, activation='relu', 
                     kernel_regularizer=regularizers.l1(l1_value))(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Definimos el modelo
    model = Model(inputs=inputs, outputs=outputs)

    # Buscamos con optuna el Learning rate
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Buscamos con optuna el pptimizador
    optimizer_options = ["adam", "sgd", "rmsprop"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_selected == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    elif optimizer_selected == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def objective(trial):
    with mlflow.start_run(nested=True):
        model = create_model(trial)

        # Buscamos con optuna Batch size
        batch_size = trial.suggest_int("batch_size", 32, 256, step=32)

        # Buscamos con optuna el número de épocas
        epochs = 10

        # Callbacks
        filepath = "best_modelconv.keras"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='loss', mode='min', restore_best_weights=True, patience=5, verbose=1)
        callbacks = [earlystop, checkpoint]

        # Entrenar sin validación
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)

        # Evaluar en test para objective
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Loggeamos hiperparámetros y métricas en MLflow
        mlflow.log_params(trial.params)
        mlflow.log_metric("loss", loss)
        mlflow.log_metric("accuracy", accuracy)

        # Loggeamos artifact
        mlflow.log_artifact(filepath, artifact_path="models")

    return loss

# Configuramos Mlflow
mlflow.set_experiment("conv1d_optuna")
# Creamos estudio en Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Número de pruebas terminadas: ", len(study.trials))

best_trial = study.best_trial
print("Mejor intento: ", best_trial)
print("Valor: ", best_trial.value)
print("Hiperparámetros: ", best_trial.params)

