import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
    
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) #En esta línea implemento la mejora para inicializar pesos
                        for x, y in zip(sizes[:-1], sizes[1:])]
        
        # Aquí voy a inicializar los momentos para los pesos y los bias, ya que me dí cuenta que si los 
        # incializaba en Update_mini_batch estaría reiniciando los valores de estos momentos en cada iteración
        # lo que hace que el efecto del algoritmo adam se pierda.
        # Algo que no explique es que los momentos se inician con la misma dimensión que la matriz
        # de pesos y la de bias, esto porque aseguramos que cada elemento de la matriz de los momentos
        # corresponda directamente a un elemento específico en los pesos/biases, esto es debido a la naturaleza el
        # optimizador Adam ya que  ajusta el tamaño del paso de actualización para cada peso y bias de forma independiente.

        self.M_w = [np.zeros(w.shape) for w in self.weights]
        self.R_w = [np.zeros(w.shape) for w in self.weights]
        self.M_b = [np.zeros(b.shape) for b in self.biases]
        self.R_b = [np.zeros(b.shape) for b in self.biases]
        self.t = 0 # Esto es un contador, que me sirve para contabilizar el número de veces que paso por un mini_batch del conjunto
        # de minibatches es decir, cuantas veces se ha llamado a la función "update_mini_batch". Este contador lo usamos en la correción
        # del bias de los momentos.



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


    def SGD(self, training_data, epochs, mini_batch_size, eta,beta_1,beta_2,
            test_data=None): # Aquí solo he actualizado el argumento de la función con los hiperparámetros beta_1, beta_2

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
                self.update_mini_batch(mini_batch, eta,beta_1,beta_2)# Aquí actualicé el argumento de la función con los hiperparámetros beta_1, beta_2
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))


    def update_mini_batch(self, mini_batch, eta, beta_1, beta_2): # Aquí actualicé el argumento de la función con los hiperparámetros beta_1, beta_2
        # El optimizador Adam lo quiero definir aquí en el update_mini_batch
        # porque en la parte del código de SGD solo definimos los datos de entrenamiento y hacemos
        # el proceso de barajear los mini batches para entrenar. 
        # Aquí en update_mini_batch es donde actualizamos los pesos y bias dependiendo
        # de los gradientes nabla_b y nabla_w que se actualizaron en backpropagation.
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Aquí hago las operaciones para los momentos, con las fórmulas vistas en clase
        self.t += 1 # El contador se actualiza por cada mini_batch que pasemos.

        # Aquí actualizo el primer y segundo momento para los pesos y bias de la red neuronal. El subindice
        # b es de bias y w es de pesos (weights).

        # Aquí uso una forma compacta de iterar con un ciclo for. Cada self y nabla contienen un array en NumPy 
        # por cada capa de la red excepto la de inputs. Este ciclo itera sobre las capas, osea, sobre los elementos 
        # de las listas self y nabla, luego en cada iteración aplica la fórmula de actualización de ADAM a cada par de arrays
        # que serían el momento anterior y el gradiente para el parámetro b y w. 
        # Lo que el zip hace es combinar las listas self y nabla en una secuencia de tuplas, donde cada tupla contiene los elementos 
        # correspondientes de cada iterable.

        self.M_b = [beta_1*mb + (1-beta_1)*db for mb, db in zip(self.M_b, nabla_b)] 
        self.R_b = [beta_2*rb + (1-beta_2)*(db**2) for rb, db in zip(self.M_b, nabla_b)]
        self.M_w = [beta_1*mw + (1-beta_1)*dw for mw, dw in zip(self.M_b, nabla_w)]
        self.R_w = [beta_2*rw + (1-beta_2)*(dw**2) for rw, dw in zip(self.M_b, nabla_w)]
        # Aquí calculo los momentos corregidos o promedios corregidos de los bias y pesos, uso
        # la misma técnica de iteración que al calcular los momentos.
        M_hatw = [mw / (1 - beta_1**self.t) for mw in self.M_w]
        R_hatw = [rw / (1 - beta_2**self.t) for rw in self.R_w]
        M_hatb = [mb / (1 - beta_1**self.t) for mb in self.M_b]
        R_hatb = [rb / (1 - beta_2**self.t) for rb in self.R_b]
        self.weights = [w-(eta/len(mini_batch))*Mw/((np.sqrt(Rw))+1e-18)
                        for w, Mw, Rw in zip(self.weights, M_hatw, R_hatw)]
        self.biases = [b-(eta/len(mini_batch))*Mb/((np.sqrt(Rb))+1e-18)
                        for b, Mb, Rb in zip(self.biases, M_hatb, R_hatb)]

        # Aquí actualizo los pesos pero
        self.weights = [w-(eta/len(mini_batch))*Mw/((np.sqrt(Rw))+1e-18)
                        for w, Mw, Rw in zip(self.weights, M_hatw, R_hatw)]
        self.biases = [b-(eta/len(mini_batch))*Mb/((np.sqrt(Rb))+1e-18)
                        for b, Mb, Rb in zip(self.biases, M_hatb, R_hatb)]

        
        

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

