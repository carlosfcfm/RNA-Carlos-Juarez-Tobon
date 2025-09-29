import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import mlflow
import os
from getpass import getpass

############### Inicializamos conectividad con el repositorio "Experimentos" en DagsHub ###########
REPO_NAME= "Experimentos"
REPO_OWNER= "carlosfcfm"  #Escribir nombre de repositorio
USER_NAME = "carlosfcfm" #Escribir su usuario
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token or password: ')
mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')
###################################################################################################

######## Construcción del modelo ####################
loss_tracker = keras.metrics.Mean(name="loss")

class Funsol(keras.Model):
    @property
    def metrics(self):
        return [loss_tracker]
    
    def train_step(self, data):
        batch_size = 300
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        eq = 3 * tf.math.sin(np.pi * x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mse_loss = keras.losses.MeanSquaredError()
            loss = mse_loss(eq, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}

inputs = keras.Input(shape=(1,))
x = keras.layers.Dense(150, activation='tanh')(inputs)
x = keras.layers.Dense(100, activation='tanh')(x)
x = keras.layers.Dense(1)(x)
model = Funsol(inputs=inputs, outputs=x)
model.summary()
###################################################################

#### Setup y entrenamiento del modelo con logging de MLflow #########################
filepath = "best_model_funsol2.keras"
mlflow.set_experiment("Funsol")

mlflow.start_run(nested=True)
mlflow.tensorflow.autolog(log_models=False)
###################################################################################


########## Compilación del modelo y entrenamiento ##########################
model.compile(optimizer=Adam(learning_rate=0.05), metrics=['loss'])
x=tf.linspace(-1,1,100)
history = model.fit(x,epochs=70,verbose=1)
###########################################################################

######### Salvamos el modelo y lo loggeamos #########
model.save(filepath)
mlflow.log_artifact(filepath, artifact_path="models")
mlflow.end_run()
######################################################

x_testv = tf.linspace(-1,1,100)
a=model.predict(x_testv)

plt.plot(x_testv,a)
plt.plot(x_testv, 3 * tf.math.sin(np.pi * x))
plt.show()

