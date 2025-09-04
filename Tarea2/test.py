import mnist_loader # Cargamos el dataset del MNIST para entrenar a la red
import network as network
import pickle #Esto será importante para guardar archivos


# Aquí cargamos los datos para entrenar, para ajustar hiperparámetros y para evaluar el rendimiento
training_data, validation_data , test_data = mnist_loader.load_data_wrapper() 

training_data = list(training_data)
test_data = list(test_data)

# Definimos las características de la red, 
# 784 neuronas de entrada (input)
# 30 neuronas en la capa media
# 10 neuronas en la capa de salida (output)
net=network.Network([784,30,10])

# Aquí definimos 15 épocas, con un mini_batch de tamaño 10 y un learning rate de 0.1
net.SGD( training_data, 15, 10, 0.1, test_data=test_data)

# Aquí abrimos el archivo ".pkl" generado al finalizar el entrenamiento y lo guardamos
archivo = open("red_prueba.pkl",'wb') 
pickle.dump(net,archivo) 
archivo.close() 
exit()
