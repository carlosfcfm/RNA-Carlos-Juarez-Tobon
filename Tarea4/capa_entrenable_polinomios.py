import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from matplotlib import pyplot as plt
import numpy as np
import math

loss_tracker = keras.metrics.Mean(name="loss")

class Funsol(keras.Model):
    @property
    def metrics(self):
        return [loss_tracker] #igual cambia el loss_tracker

    def train_step(self, data):
        batch_size =10 #Calibra la resolucion de la ec.dif
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        eq = tf.math.cos(2*x) 

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mse_loss = keras.losses.MeanSquaredError()
            loss = mse_loss(eq, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        #actualiza metricas
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}
    
class SinTransform(tf.keras.layers.Layer):
    def __init__(self, max_potencia=3):
        super(SinTransform,self).__init__()
        self.max_potencia = max_potencia

        self.kernel = self.add_weight(
        shape=[self.max_potencia])

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if (inputs.shape == ()):
            inputs=(inputs,)
        elif (len(inputs.shape)==1):
            inputs=tf.expand_dims(inputs, axis=1)
        batch = tf.shape(inputs)[0]
        potencias = tf.range(0, self.max_potencia + 1, dtype=tf.float32)
        potencias = tf.expand_dims(potencias, axis=0)
        x_potencias = tf.pow(inputs, potencias)

        return tf.tensordot(x_potencias,self.kernel,axes=[[-1], [0]])
    
trans = SinTransform(3)
x = tf.random.uniform((3,), minval=-1, maxval=1)
print(x)
res=trans(x)
print(res)


inputs = keras.Input(shape=(1,))
print(inputs)
x = SinTransform(3)(inputs)
model = Funsol(inputs=inputs,outputs=x)
model.summary()


model.compile(optimizer=SGD(learning_rate=0.01), metrics=['loss'])

x=tf.linspace(-10,10,100)
history = model.fit(x,epochs=100,verbose=1)


x_testv = tf.linspace(-10,10,100)
a=model.predict(x_testv)

plt.plot(x_testv,a)
plt.plot(x_testv, tf.math.cos(2*x) )
plt.show()