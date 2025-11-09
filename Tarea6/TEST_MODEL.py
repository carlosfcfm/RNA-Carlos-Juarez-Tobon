import tensorflow as tf
from tensorflow import keras
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
# 1. RUTAS Y CONFIG
# -------------------------------
BASE_DIR = r"C:\Users\LamdaZero\Desktop\Concurso2\plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -------------------------------
# 2. CARGAR DATOS Y SPLIT 
# -------------------------------
df = pd.read_csv(TRAIN_CSV)
df['image_id'] = df['image_id'] + '.jpg'
df = df.drop_duplicates(subset='image_id').reset_index(drop=True)

labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
df['class'] = np.argmax(df[labels].values, axis=1)

# Split: 30% test
df_train_val, df_test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df['class'])
df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=SEED, stratify=df_train_val['class'])

print(f"Test size: {len(df_test)} imágenes")

# -------------------------------
# 3. DATASET DE TEST
# -------------------------------
def Creador_datasets_test(df_split):
    file_paths = [os.path.join(IMG_DIR, img_id) for img_id in df_split['image_id']]
    labels = df_split['class'].values
    
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    def pre_procesamiento(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, tf.one_hot(label, 4)
    
    ds = ds.map(pre_procesamiento, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = Creador_datasets_test(df_test)

# -------------------------------
# 4. CARGAMOS EL MODELO .keras
# -------------------------------
MODEL_PATH = "best_effnet_finetuned_final.keras"  

model = keras.models.load_model(MODEL_PATH)

# -------------------------------
# 5. EVALUACIÓN EN TEST
# -------------------------------
print("\nEvaluamos en el conjunto de TEST")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nRESULTADOS EN TEST:")
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# -------------------------------
# 6. LOG EN MLFLOW
# -------------------------------
with mlflow.start_run(run_name="Evaluacion_Final_Test"):
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log modelo (opcional)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model_final")

