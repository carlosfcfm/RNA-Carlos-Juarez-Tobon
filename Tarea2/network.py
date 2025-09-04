import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) #En esta línea implemento la mejora para inicializar pesos
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # Originalmente inicializamos los pesos en "self.weights" con números aleatorios que siguen una distribución
        # Gaussiana con media 0 y desviación estándar 1. El problema de esto es que las neuronas se pueden saturar muy seguido,
        # esa saturación implica que las neuronas dejan de aprender. Durante backpropagation, los gradientes multiplican las derivadas de cada 
        # neurona y si muchas neuronas están saturadas, los gradientes se vuelven muy pequeños, así que se estanca el aprendizaje.
        # Nos evitamos este problema al inicializar los pesos con números aleatorios que siguen una distribución Gaussiana con media 0 pero
        # desviación estándar 1/sqrt(x) donde sqrt(x) es la raíz del número de neuronas de entrada (input). La varianza ahora es pequeña y con esto
        # cuando las neuronas calculen las z's, estas no serán muy grandes ni muy pequeñas al inicio, evitando el problema del gradiente.

        #Lo único que hago en self.weights es que al hacer el array de pesos, a cada elemento del array
        # lo divido por la raíz del número de inputs (entradas) de la primera capa. Así hemos modificado 
        # la desviación estándar de esos números con "np.random.randn(y, x)/np.sqrt(x)"
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):

        training_data = list(training_data)
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))


    def update_mini_batch(self, mini_batch, eta):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        

    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x] 

        zs = [] 
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)



        delta = self.Cross_Entropy(activations[-1], y) #Aquí implementé la función de costos Cross-Entropy, como tal no es más que un cambio al cálculo de delta.
        # La cross-entropy lo que hace es eliminar la derivada de la función sigmoide, por eso es poderosa ya que al eliminar esta derivada
        # ya no dependerá de este factor y por lo tanto ya no "aprende lento", ahora solo depende del error delta.
        # Sin embargo, las capas ocultas aún dependen de las derivadas de la sigmoide y los pesos para propagar el error, por eso solo esta
        # línea de código se actualiza


        

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) 

        for l in range(2, self.num_layers):

            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) 



    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) 

    def Cross_Entropy(self, output_activations, y):
        ### Esta función hace lo mismo que hacía "cost_derivative", solo que la renombré
        ### para dejar en claro que Cross_Entropy la podemos colocar como una función
        ### de python

        return (output_activations-y)

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

