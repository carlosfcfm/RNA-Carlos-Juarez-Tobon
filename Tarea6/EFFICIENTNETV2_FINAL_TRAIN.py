import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import mlflow
import getpass

# ------------------- CONFIGURACIÓN MLFLOW -------------------
REPO_OWNER = "carlosfcfm"
REPO_NAME = "Experimentos"
USER_NAME = "carlosfcfm"

os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass.getpass('Enter your DAGsHub access token: ')

mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
mlflow.set_experiment("T6_EfficientNetV2M_FINAL")

# -------------------------------
# 1. CONFIGURACIÓN
# -------------------------------
BASE_DIR = r"C:\Users\LamdaZero\Desktop\Concurso2\plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# 2. CARGAMOS Y DIVIDIMOS LOS DATOS
# -------------------------------
df = pd.read_csv(TRAIN_CSV)
df['image_id'] = df['image_id'] + '.jpg'
df = df.drop_duplicates(subset='image_id').reset_index(drop=True)

labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
df['class'] = np.argmax(df[labels].values, axis=1)

# Split: 30% test, 70% → train/val (56%/14%)
df_train_val, df_test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['class'])
df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=SEED, stratify=df_train_val['class'])

total = len(df)
print(f"Total: {total}")
print(f"Train: {len(df_train)} ({len(df_train)/total:.1%})")
print(f"Val:   {len(df_val)} ({len(df_val)/total:.1%})")
print(f"Test:  {len(df_test)} ({len(df_test)/total:.1%})")

# -------------------------------
# 4. DATASETS 
# -------------------------------
def Creador_datasets(df_split, shuffle=False, batch_size=BATCH_SIZE):
    file_paths = [os.path.join(IMG_DIR, img_id) for img_id in df_split['image_id']]
    labels = df_split['class'].values
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def pre_procesamiento(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, tf.one_hot(label, 4)
    
    ds = ds.map(pre_procesamiento, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = Creador_datasets(df_train, shuffle=True)
val_ds = Creador_datasets(df_val, shuffle=False)
test_ds = Creador_datasets(df_test, shuffle=False)

# -------------------------------
# 5. MEJORES HIPERPARÁMETROS DE OPTUNA 
# -------------------------------
mejores_parametros = {
    'lr': 0.0004582697234246692,
    'dropout1': 0.3490308554014889,
    'dropout2': 0.22737876634660156,
    'dropout3': 0.21851365881263013,
    'dropout4': 0.2804592098090995,
    'units1': 2396,
    'units2': 1685,
    'units3': 568,
    'units4': 142,
    'rotation': 0.26458245928892987,
    'zoom': 0.13275608618307683,
    'contrast': 0.35573651795425903
}

# -------------------------------
# 6. CREAMOS EL MODELO
# -------------------------------
base_model = keras.applications.EfficientNetV2M(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  

# Data augmentation con hiperparámetros
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(mejores_parametros['rotation']),
    layers.RandomZoom(mejores_parametros['zoom']),
    layers.RandomContrast(mejores_parametros['contrast']),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="aug")

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)
x = keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

# Capas densas con dropout
x = layers.Dense(mejores_parametros['units1'], activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(mejores_parametros['dropout1'])(x)

x = layers.Dense(mejores_parametros['units2'], activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(mejores_parametros['dropout2'])(x)

x = layers.Dense(mejores_parametros['units3'], activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(mejores_parametros['dropout3'])(x)

x = layers.Dense(mejores_parametros['units4'], activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(mejores_parametros['dropout4'])(x)

outputs = layers.Dense(4, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# Callbacks
filepath = "best_effnet_finetuned.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
callbacks_list = [checkpoint, earlystop]

# -------------------------------
# 7. MLFLOW: FASE 1 + FASE 2 
# -------------------------------
with mlflow.start_run(run_name="EfficientNetV2M_Entrenamiento_Completo") as parent_run:
    mlflow.tensorflow.autolog(log_models=False, silent=True)
    mlflow.log_params(mejores_parametros)
    
    print("FASE 1: ENTRENAMIENTO DEL CLASIFICADOR ")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=mejores_parametros['lr']),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        callbacks=callbacks_list,
        verbose=1
    )

    # Guardamos el modelo de la Fase 1
    model_fase1_path = "model_fase1_effnet.keras"
    model.save(model_fase1_path)
    mlflow.log_artifact(model_fase1_path, artifact_path="models")

    # === FASE 2: FINE-TUNING  ===
    with mlflow.start_run(nested=True, run_name="Fine_Tuning_EfficientNetV2M") as child_run:
        mlflow.tensorflow.autolog(log_models=False, silent=True)
        print("FASE 2: FINE-TUNING")

        # Descongelamos las últimas 50 capas
        base_model.trainable = True
        for layer in base_model.layers[:-50]:
            layer.trainable = False

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=callbacks_list,
            verbose=1
        )
        # Guardamos el modelo final
        model_final_path = "best_effnet_finetuned_final.keras"
        model.save(model_final_path)
        mlflow.log_artifact(model_final_path, artifact_path="models")