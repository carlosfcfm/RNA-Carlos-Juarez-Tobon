import pandas as pd
import mlflow
import os
from getpass import getpass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
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
# Hacemos el reshape para CNN 1D con forma: (muestras, características, 1 canal)
X_train = X_train.reshape((X_train.shape[0], num_features, 1))
X_test = X_test.reshape((X_test.shape[0], num_features, 1))
X_val = X_val.reshape((X_val.shape[0], num_features, 1))

# Hiperparámetros del modelo original (para logging)
hyperparameters = {
    "filters_1": 128,
    "filters_2": 192,
    "filters_3": 384,
    "kernel_size": 3,
    "dense_units": 512,
    "l1_value": 2.1179368863512548e-05 ,
    "learning_rate": 4.808589316209705e-05,
    "batch_size": 64,
    "epochs": 40,
    "optimizer": "adam"
}

# Cargamos el modelo guardado
model_path = r"C:\Users\LamdaZero\Desktop\Pruebas 2\best_modelconv_l1.keras"
model = keras.models.load_model(model_path)

# Configuramos MLflow
mlflow.set_experiment("Tarea5_Modelo_validacion")
with mlflow.start_run(run_name="Validacion Mejor Modelo"):
    mlflow.tensorflow.autolog(log_models=False)

    # Loggeamos hiperparámetros del modelo original
    mlflow.log_params(hyperparameters)

    # Evaluamos en validación
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)
    print(f"Precisión en validación: {val_accuracy:.4f}")
    print(f"Pérdida en validación: {val_loss:.4f}")

