import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import os
from getpass import getpass

###### RESUMEN DE LOS TRES EXPERIMENTOS #######

# Aquí solo añado los cambios realizados respecto a la versión
# o experimento anterior.  

#Experimento 1 ajustes: #
# batch_size = 12       #
# Optimizador : Adam #
# Learning rate: 0.0003 #
# Capa 2: 40 neuronas #
# Capa 3: 20 neuronas + relu#
# Función de costos: categorical_crossentropy#


#Experimento 2 ajustes: #
# batch_size = 20       #
# Optimizador : RMSprop #
# Learning rate: 0.001 #
# Capa 3: 30 neuronas + relu #


#Experimento 3 #
# batch_size = 40   #
# Optimizador : Adam #
# Learning rate: 0.00025 #
# Capa 2: 50 neuronas + relu #
# Capa 3: 30 neuronas + relu#

##################################################################

############### Inicializamos conectividad con el repositorio "Experimentos" en DagsHub ###########
REPO_NAME= "Experimentos"
REPO_OWNER= "carlosfcfm"  #Escribir nombre de repositorio
USER_NAME = "carlosfcfm" #Escribir su usuario
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token or password: ')
mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
dataset=mnist.load_data()
###################################################################################################

####################### Cargamos datos del mnist y aplanamos los datos de entrenamiento y de test ###
(x_train, y_train), (x_test, y_test) = dataset

x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)
x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')

x_trainv /= 255.  # x_trainv = x_trainv/255
x_testv /= 255.

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)
####################################################################################################

############# Diseño de la red #################################
model = Sequential()
model.add(Dense(50, activation='relu', input_shape=(784,))) # Capa 1 (Input) + Capa 2 (oculta)
model.add(Dense(30, activation='relu')) # Capa 3
model.add(Dense(num_classes, activation='softmax')) # Capa 4 Output
model.summary()
###############################################################



#### Setup y entrenamiento del modelo con logging de MLflow y callbacks ################################

filepath = "best_model_exp3.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True, patience=15,verbose=1)
mlflow.set_experiment("experimentos_sin_regularizacion")

mlflow.start_run(nested=True)
mlflow.tensorflow.autolog(log_models=False)
model.compile(loss="categorical_crossentropy",optimizer=Adam(learning_rate=0.00025),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size = 40,
                    epochs = 15,
                    verbose=1,
                    validation_data=(x_testv, y_testc),

                    callbacks=[earlystop, checkpoint])
model.save(filepath)
mlflow.log_artifact(filepath, artifact_path="models")
mlflow.end_run()
##########################################################################################################


score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)

