import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.keras


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.autolog()

dataset=mnist.load_data()

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


model = Sequential()
model.add(Dense(40, activation='sigmoid', input_shape=(784,)))
model.add(Dense(20, activation='relu'))
model.add(Dense(num_classes, activation='softmax')) 
model.summary()

filepath = "best_model.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystop = EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True, patience=10,verbose=1)
mlflow.set_experiment("experimento1")

mlflow.start_run(nested='TRUE')
model.compile(loss="categorical_crossentropy",optimizer=Adam(learning_rate=0.0005),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size = 12,
                    epochs = 15,
                    verbose=1,
                    validation_data=(x_testv, y_testc),

                    callbacks=[earlystop, checkpoint])
mlflow.keras.save_model(model, "best_model")
mlflow.log_artifact("best_model", artifact_path="models")
mlflow.end_run()


score = model.evaluate(x_testv, y_testc, verbose=1) #evaluar la eficiencia del modelo
print(score)