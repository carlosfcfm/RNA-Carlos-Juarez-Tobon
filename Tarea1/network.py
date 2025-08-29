#En este código se implementa una red neuronal con el descenso del gradiente y backpropagation.
#Esta red la usamos para reconocer imágenes de números del 0 al 9.
import random
import numpy as np

# Utilizamos "class" que es un comando utilizado en programación orientado a objetos y este define
# las propiedades y funciones de un objeto.
# En este caso definimos la clase "Network" como un objeto
class Network(object):
    # Usamos esta función "_init_" que lo podemos ver como un manual de instrucciones
    # para el objeto que acabamos de crear que es "Network". Aquí se configura la
    # red, definimos capas, pesos, bias, que es lo escencial para que funcione la red.
    def __init__(self, sizes):
    
        # "Sizes" es una lista que contiene el número de neuronas en cada capa de la red. En este caso
        # particular tenemos una red de "[784,30,10]", 3 capas, con la primera capa siendo 784 entradas
        # que serían los pixeles de una imagen de 28x28, tenemos 30 neuronas en la capa 2 y 10 neuronas
        # de salida en la capa 3, que sería un vector de 10 entradas.
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # Aquí los biases y weights inician con valores aleatorios con una distribución Gaussiana.
        # Aquí también tenemos una lista de matrices de biases, no considera la primera capa porque esos son solo datos
        # de entrada, así que por eso agarramos la segunda capa en adelante "[1:]", el tamaño de esta
        # matriz esta definida con (y,1), con "y" siendo el número de neuronas de la capa correspondiente
        # vemos que sería un vector columna.
        # Para los pesos, tenemos igualmente otra lista de matrices, pero esta es de tamaño (y,x)
        # y son las neuronas de la capa siguiente y x las de la capa anterior ya que a cada neurona
        # de la capa siguiente le tocan los pesos de las neuronas de la capa anterior


# Con "feedforward" usamos los datos de entrada y aquí producimos una salida con la función de activación sigmoide. 
# Itera sobre los pares de bias b y pesos w de cada capa y hacemos un producto matricial entre los pesos y 
# el vector de entrada y sumamos los bias. La función sigmoide sirve para "normalizar" los valores de la salida 
# entre 0 y 1 y al final conseguimos una activación de las neuronas y esto lo vemos como un vector de activaciones.
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


########## MÉTODO SGD #############
# Usamos el método del descenso del gradiente estocástico. Este método nos ayuda a encontrar el mínimo
# en la función de costos, para hacer predicciones más precisas, lo cual también significa minimizar los w's y b's.
# Con este método usamos solo un suconjunto de los datos de entrenamiento "mini-batch" y generamos "mini-batches" del training_data,
# cada minibatch tiene un vector de entrada de 784 elementos y otro vector, representando una etiqueta que nos dice que dígito
# entre el 0 y 9 le corresponde a ese vector de entrada es decir (el número correcto que se puede ver en la imagen).

# Entonces generamos un conjunto de subconjuntos (mini_batches) de los datos de entrenamiento 
# y al terminar de procesar todos los datos de entrenamiento, 
# se completa una época, y se actualizan los pesos y los biases. Repetimos este ciclo
# hasta las tantas épocas que decidamos entrenar a la red.

# El learning rate "eta" nos define el tamaño de las actualizaciones de los pesos y sesgos durante el entrenamiento.
# En términos sencillos, define que tan grandes son los pasos del gradiente para bajar al mínimo de la función de costos
# Si es grande el learning rate, el gradiente baja y sube sin control, si es pequeño se estanca en mínimos locales.


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
            # Se puede ver que para cada iteración o (época) se mezclan los datos de entrenamiento
            # con "random.shuffle" para evitar sesgar a la red por el orden que tengan.
            # En esta parte del código es donde creamos los mini_batches, divide los datos de entrenamiento
            # en sublistas de tamaño "mini_batch_size", si hay 1000 datos y el tamaño del minibatch quiero que sea
            # 10, entonces tendría 100 mini batches.

    def update_mini_batch(self, mini_batch, eta):
        # Actualizamos los pesos y sesgos de la red para un mini-batch usando gradientes 
        # calculados por retropropagación. Para cada mini-batch, el método backprop calcula los gradientes (nabla_b y nabla_w)
        # que indican en qué dirección y cuánto deben ajustarse los pesos y sesgos 
        # Este algoritmo de retropropagación o backpropagation lo explicaré abajo cuando definamos la función.

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # El x representa el vector de entrada en la red, "y" es un vector one-hot de 10 elementos para indicar el dígito correcto.
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        

#### ALGORITMO BACKPROPAGATION #####
# Este algoritmo toma inspiración de que al calcular los gradientes de la función de costos respecto de w y b, se empieza desde la útlima capa
# y todas las derivadas se calculan hacia atrás con ayuda de la regla de la cadena, por eso es una "retropropagación". Esto sirve para ajustar
# los w's y b's automáticamente y reducir el error en la siguiente predicción. Al final devuelve los gradientes de la función de costo respecto 
# a los sesgos (nabla_b) y pesos (nabla_w). Estos gradientes indican la dirección y magnitud en la que deben ajustarse los parámetros 
# para minimizar el error.

    def backprop(self, x, y):

        # Generamos listas de matrices de ceros y aquí guardamos los resultados más adelante
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # lista que guarda todas las activaciones capa a capa

        zs = [] 
        # Este bucle realiza el feedforward, calculando activaciones capa por capa.
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass



        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # Calcula el error (delta) en la capa de salida: derivada de la de costo respecto a la salida 
        # multiplicado por la derivada de la función sigmoide respecto de las z's. "sigmoid_prime" ajusta el error 
        # para reflejar cuánto afecta un cambio en z^L al error total 
        # z^L representa los valores en crudo de la última capa antes de pasar a la sigmoide

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) # Calcula el gradiente de los pesos de la última capa

        for l in range(2, self.num_layers):
                # Este ciclo retrocede desde la penúltima capa hasta la primera capa oculta 
                # (usando índices negativos: l=2 es la segunda capa desde el final y así sucesivamente
                # Aquí es donde propagamos el error hacia atrás capa por capa
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w) #Estos gradientes se usan en "update_mini_batch" para ajustar los parámetros con SGD


    #### Función evaluate ####
    # Esta función solo devuelve la evaluación de la precisión de la red. 
    # Se utiliza np.argmax para obtener el índice de la neurona con mayor activación en la capa de salida 
    # esto le permite a la red clasificar los dígitos mostrados en los datos de entrenamiento y determinar un número entre
    # 0 y 9
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results) ## Compara la predicción con la etiqueta "y" y suma las coincidencias correctas.

    # La función "cost_derivative" solo calcula la función de costos respecto a las activaciones de la capa de salida
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x /
        partial a for the output activations."""

        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
