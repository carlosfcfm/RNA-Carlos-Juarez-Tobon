import mnist_loader
import network as network
import pickle

training_data, validation_data , test_data = mnist_loader.load_data_wrapper()

training_data = list(training_data)
test_data = list(test_data)

net=network.Network([784,30,10])

net.SGD( training_data, 15, 10, 0.9, test_data=test_data)

archivo = open("red_prueba.pkl",'wb') 
pickle.dump(net,archivo) 
archivo.close() 
exit()
