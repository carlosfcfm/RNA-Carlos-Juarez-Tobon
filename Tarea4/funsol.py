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
        return [loss_tracker]
    
    def train_step(self, data):
        batch_size = 10
        x = tf.random.uniform((batch_size,), minval=-1, maxval=1)
        eq = 3 * tf.math.sin(np.pi* x)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            mse_loss = keras.losses.MeanSquaredError()
            loss = mse_loss(eq, y_pred)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        loss_tracker.update_state(loss)

        return {"loss": loss_tracker.result()}

class SinTransform(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(SinTransform, self).__init__()
        self.num_outputs = int(num_outputs)

        self.freq = tf.range(1, self.num_outputs + 1, dtype=tf.float32)

        self.kernel = self.add_weight(shape=[self.num_outputs])

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        if (inputs.shape == ()):
           inputs=(inputs,)
        elif (len(inputs.shape) == 1):
            inputs = tf.expand_dims(inputs, axis=1)
        batch = tf.shape(inputs)[0]
        freq_b = tf.ones([batch, 1]) * self.freq
        args = freq_b * inputs
        modes = tf.math.sin(args)
        return tf.tensordot(modes, self.kernel, 1)
    
trans = SinTransform(5)
x = tf.random.uniform((3,), minval=-1, maxval=1)
print("Input:", x)
res = trans(x)
print("Output:", res)

inputs = keras.Input(shape=(1,))
print(inputs)
x = SinTransform(5)(inputs)
model = Funsol(inputs=inputs,outputs=x)
model.summary()

model.compile(optimizer=Adam(learning_rate=0.01), metrics=['loss'])
x=tf.linspace(-1,1,100)
history = model.fit(x,epochs=50,verbose=1)

x_testv = tf.linspace(-1,1,100)
a=model.predict(x_testv)

plt.plot(x_testv,a)
plt.plot(x_testv, 3 * tf.math.sin(np.pi* x_testv))
plt.show()

