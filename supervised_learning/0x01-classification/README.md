# Classification - redes neuronales - apuntes

### Testing

For test this project is need to take care of the binary files

``` 
# execute this commnad
$ ./show_data.py
```

![](https://i.imgur.com/N3Fmydu.png)

# Concepts - notes

### Perceptrones

el modelo de neurona principal utilizado es uno llamado **neurona sigmoidea**, Un perceptrón toma varias entradas binarias, X1, X2, …y produce una única salida binaria. La salida de la neurona, 0 o 1, se determina por si la suma ponderada, el umbral es un número real que es un parámetro de la neurona

Otra forma perceptrones se pueden utilizar es calcular las funciones lógicas elementales que normalmente consideramos como computación subyacente, funciones tales como AND, OR, y NAND.

![](https://i.imgur.com/AKvMl6p.png)
![](https://i.imgur.com/nsgFf5l.png)

producto escalar, w ⋅ x, dónde w y Xson vectores cuyos componentes son los pesos y las entradas, respectivamente, el umbral al otro lado de la desigualdad y reemplazarlo por lo que se conoce como sesgo del perceptrón , b ≡ - umbral

suppose we did have a perceptron with no inputs. Then the weighted sum ∑jwjxj would always be zero, and so the perceptron would output 1 if b>0, and 0 if b≤0. That is, the perceptron would simply output a fixed value, not the desired value (x1, in the example above).
![](https://i.imgur.com/lgHiXhv.png)
 perceptrones sin entrada se toman más bien unidades especiales que simplemente se definen para generar los valores deseados, X1, X2

``` 
b = sesgo del perseptron
```

El sesgo se puede tomar como una medida de lo fácil que es lograr que el perceptrón emita un 1. O para decirlo en términos más biológicos, el sesgo es una medida de lo fácil que es hacer que el perceptrón se dispare . Para un perceptrón con un sesgo realmente grande, es extremadamente fácil para el perceptrón generar un1. Pero si el sesgo es muy negativo, entonces es difícil para el perceptrón generar un1.

![](https://i.imgur.com/9L0l0Sz.png)

![](https://i.imgur.com/NTAfElO.png)

En esta red, la primera columna de perceptrones, lo que llamaremos la primera capa de perceptrones, está tomando tres decisiones muy simples, sopesando la evidencia de entrada.

* ![](https://i.imgur.com/s1KK3jO.png)
* ![](https://i.imgur.com/ZWU2sGn.png)

----

## funcion es de activación

calcula una "suma ponderada" de su entrada, agrega un sesgo y luego decide si debe "dispararse" o no

* Función de paso

Una función paso es una función definida por partes en la cual cada parte es un segmento de recta horizontal o un punto.
 
![](https://i.imgur.com/NuVRoah.png)

### Función lineal

A = cx, la derivada con respecto a x es c. Eso significa que el gradiente no tiene relación con X. Es un gradiente constante y el descenso será en gradiente constante. Si hay un error en la predicción, los cambios realizados por retropropagación son constantes y no dependen del cambio en el delta de entrada (x).

No importa cuántas capas tengamos, si todas son de naturaleza lineal, la función de activación final de la última capa no es más que una función lineal de la entrada de la primera capa. Haga una pausa un poco y piénselo.

### Función Tanh

es una función sigmoidea escalada. tan genial que podemos apilar capas! Está destinado al rango (-1, 1), por lo que no se preocupe de que las activaciones exploten. Un punto a mencionar es que el gradiente es más fuerte para tanh que para sigmoide
![](https://i.imgur.com/DqgsDyP.png)

### funcion ReLu

El rango de ReLu es [0, inf). Esto significa que puede hacer estallar la activación. ReLu nos da este beneficio. Imagine una red con pesos inicializados aleatorios (o normalizados) y casi el 50% de la red produce activación 0 debido a la característica de ReLu (salida 0 para valores negativos de x). Esto significa que se están disparando menos neuronas (activación escasa) y la red es más ligera.

``` 
A (x) = máximo (0, x)
```

ReLu moribundo. Este problema puede hacer que varias neuronas simplemente mueran y no respondan, lo que hace que una parte sustancial de la red sea pasiva. Existen variaciones en ReLu para mitigar este problema simplemente convirtiendo la línea horizontal en un componente no horizontal.

### Neuronas sigmoideas

Las neuronas sigmoides son similares a los perceptrones, pero modificadas de modo que pequeños cambios en sus pesos y sesgos provocan solo un pequeño cambio en su producción. Ese es el hecho crucial que permitirá que una red de neuronas sigmoides aprenda.

Al igual que un perceptrón, la neurona sigmoidea tiene entradas, X1, X2, …. Pero en lugar de ser solo0 o 1, estas entradas también pueden tomar cualquier valor entre 0 y 1, Entonces, por ejemplo, 0, 638 …es una entrada válida para una neurona sigmoidea. Además, al igual que un perceptrón
<img src="https://i.imgur.com/kcpY3d6.png" width="425"/> <img src="https://i.imgur.com/Nu7Adve.png" width="425"/> 

<center><h1> σ </h1></center>

De hecho, cuando w ⋅ x+ b = 0 las salidas del perceptrón 0, mientras que la función paso a paso 1. Entonces, estrictamente hablando, necesitaríamos modificar la función de paso en ese punto.

``` 
Una función paso es una función definida
por partes en la cual cada parte es un
segmento de recta horizontal o un punto.
```

Si σ de hecho había sido una función de paso, a continuación, la neurona sigmoide sería ser un perceptrón, puesto que la salida sería1 o 0 dependiendo de si w ⋅ x+ b fue positivo o negativo. Utilizando el actualσfunción obtenemos, como ya se indicó anteriormente, un perceptrón suavizado. De hecho, es la suavidad delσfunción que es el hecho crucial, no su forma detallada. La suavidad deσ significa que pequeños cambios Δwj en los pesos y Δ b en el sesgo producirá un pequeño cambio Δ salidaen la salida de la neurona. De hecho, el cálculo nos dice queΔ salida está bien aproximado por

![](https://i.imgur.com/Fd9Id4Q.png)
Δ salida es una función lineal de los cambiosΔwj y Δ ben los pesos y el sesgo, Esta linealidad facilita la elección de pequeños cambios en los pesos y sesgos para lograr cualquier pequeño cambio deseado en la salida.

donde la suma está sobre todos los pesos, wj, y ∂salida / ∂wj y ∂salida / ∂b denotar derivadas parciales de la producción con respecto a wj y B, respectivamente.

![](https://i.imgur.com/45bLrEX.png)

Entonces, si bien las neuronas sigmoides tienen gran parte del mismo comportamiento cualitativo que los perceptrones, hacen que sea mucho más fácil descubrir cómo cambiar los pesos y los sesgos cambiará la salida.

Lo principal que cambia cuando usamos una función de activación diferente es que los valores particulares para las derivadas parciales en la Ecuación cambian. Resulta que cuando calculamos esas derivadas parciales más tarde, usandoσsimplificará el álgebra, simplemente porque las exponenciales tienen propiedades hermosas cuando se diferencian. En todo caso, σ se usa comúnmente en el trabajo sobre redes neuronales y es la función de activación.

### Cual funcion de activacion se escoje?

Cuando sepa que la función que está tratando de aproximar tiene ciertas características, puede elegir una función de activación que aproximará la función más rápidamente y conducirá a un proceso de entrenamiento más rápido. Por ejemplo, un sigmoide funciona bien para un clasificado.

Estas funciones si uno no le parecen puede mejorarlas

### Propagación hacia adelante y hacia atras en redes neuronales

* Adelantado

Propagar los cálculos de todas las neuronas dentro de todas las capas moviéndose de izquierda a derecha. Esto comienza con la alimentación de sus vectores / tensores de características en la capa de entrada y termina con la predicción final generada por la capa de salida. Los cálculos de pase hacia adelante ocurren durante el entrenamiento para evaluar la función objetivo / pérdida bajo la configuración actual de los parámetros de red en cada iteración, así como durante la inferencia (predicción después del entrenamiento) cuando se aplica a datos nuevos / no vistos.

    En otras palabras, es donde daría una cierta entrada a su red neuronal, digamos una imagen o texto. La red calculará la salida propagando la señal de entrada a través de sus capas. En otras palabras, la salida de una capa se convierte en la entrada a la siguiente, donde la salida de la última es la “respuesta”.

* Atrás

Conocido como retropropagación , o "backprop", este es un paso que se ejecuta durante el entrenamiento para calcular el gradiente de la función objetivo / pérdida con respecto a los parámetros de la red para actualizarlos durante una sola iteración de alguna forma de descenso de gradiente (Adam, RMSProp, etc.). Se llama así porque, cuando se ve una red neuronal como un gráfico de cálculo, comienza calculando las derivadas de la función de objetivo / pérdida en la capa de salida y las propaga hacia la capa de entrada (efectivamente, esta es la regla de la cadena  de Cálculo en acción) para calcular derivadas y actualizar todos los parámetros en todas las capas.

    durante el entrenamiento es necesario actualizar (“optimizar”) todos los parámetros de las capas de la red. Por esta razón, la red necesita saber en qué "dirección" debe ir la actualización, por lo que debe calcular el llamado gradiente con respecto a una función conocida como función de pérdida, que es una forma de decir qué tan "incorrecta" o “Incorrecta” la red todavía lo es, así que espero que mejore la próxima vez.

    Dado que encontrar los gradientes de todos los parámetros con respecto a esta función de pérdida es una ecuación compleja que involucra la llamada "regla de la cadena", en el cálculo se está propagando, donde cada capa agrega su propia contribución a ese gradiente, comenzando desde la última (de ahí el nombre).

**Ejemplo**
Digamos que queremos entrenar esta red neuronal para predecir si el mercado subirá o bajará. Para ello, asignamos dos clases Clase 0 y Clase 1.

![](https://i.imgur.com/LE9hNBw.png)

###  retropropagación

![](https://i.imgur.com/RwoQZxL.png)
![](https://i.imgur.com/5entTJU.png)

### funcion de perdida

definen un objetivo con el que se evalúa el rendimiento del modelo y los parámetros aprendidos por el modelo se determinan minimizando una función de pérdida elegida.

calcular qué tan buenas son nuestras predicciones. La función de pérdida es la diferencia entre nuestros valores predichos y reales. Creamos una función de pérdida para encontrar los mínimos de esa función para optimizar nuestro modelo y mejorar la precisión de nuestra predicción.

Las funciones de pérdida definen qué es y qué no es una buena predicción. En resumen, elegir la función de pérdida correcta determina qué tan bien estará su estimador.

``` 
Pérdida = Suma (Previsto - Real)²
```

![](https://i.imgur.com/HuvrIYd.png)

Nuestro objetivo es reducir la pérdida cambiando los pesos de manera que la pérdida converja al valor más bajo posible. Intentamos reducir la pérdida de forma controlada, dando pequeños pasos hacia la pérdida mínima. Este proceso se llama Gradient Descent (GD). Mientras realizamos GD, necesitamos saber la dirección en la que deben moverse los pesos. En otras palabras, debemos decidir si aumentar o disminuir los pesos. Para conocer esta dirección, debemos tomar la derivada de nuestra función de pérdida. Esto nos da la dirección del cambio de nuestra función.

miden qué tan lejos está un valor estimado de su valor real. Una función de pérdida asigna decisiones a sus costos asociados.

 Las funciones de pérdida no son fijas, cambian en función de la tarea a realizar y del objetivo a cumplir.:

* regresión
* clasificación

### cost function

# La arquitectura de las redes neuronales

redes de múltiples capas a veces se denominan perceptrones de múltiples capas o MLP 

la salida de una capa se usa como entrada a la siguiente. Estas redes se denominan redes neuronales de retroalimentación.

**la capa oculta consta de dos funciones:**

* Función de preactivación: La suma ponderada de las entradas se calcula en esta función.
* Función de activación: Aquí, según la suma ponderada, se aplica una función de activación para hacer que la red no sea lineal y hacer que aprenda a medida que avanza el cálculo. La función de activación utiliza el sesgo para que no sea lineal.

La capa intermedia se llama capa oculta., ya que las neuronas de esta capa no son ni entradas ni salidas. 
modelos de redes neuronales artificiales en los que son posibles los bucles de retroalimentación. Estos modelos se denominan redes neuronales recurrentes. 

![](https://i.imgur.com/rYxYKxH.png)

La capa intermedia se llama capa oculta., ya que las neuronas de esta capa no son ni entradas ni salidas. 

### Artificial Neural Network - ANN

Is a type of neural network which is based on a Feed-Forward strategy. It is called this because they pass information through the nodes continuously till it reaches the output node. This is also known as the simplest type of neural network.

Some advantages of ANN :

* Ability to learn irrespective of the type of data (Linear or Non-Linear).
* ANN is highly volatile and serves best in financial time series forecasting.

Some disadvantages of ANN :

* The simplest architecture makes it difficult to explain the behavior of the network.
* This network is dependent on hardware.

### Biological Neural Network - BNN

Is a structure that consists of Synapse, dendrites, cell body, and axon. In this neural network, the processing is carried out by neurons. Dendrites receive signals from other neurons, Soma sums all the incoming signals and axon transmits the signals to other cells.

Some advantages of BNN :

* The synapses are the input processing element.
* It is able to process highly complex parallel inputs.

Some disadvantages of BNN :

* There is no controlling mechanism.
* Speed of processing is slow being it complex.

![](https://i.imgur.com/KbgxDS5.png)

### "feedforward"

Conectividad Cada conjunto de neuronas de cubo fluye hacia un conjunto diferente y no hay "reflujo"
![](https://i.imgur.com/Z8zWirL.png)

apuntes tomados de: 

* [ ] http://neuralnetworksanddeeplearning.com/chap1.html
* [ ] https://www.section.io/engineering-education/understanding-loss-functions-in-machine-learning/

# Diccionario

## Modelo

Un modelo es una representación intencional de algún sistema real Starfield et al. 1990. Construimos y usamos modelos para resolver problemas o responder preguntas sobre un sistema o una clase de sistemas. En ciencia, generalmente se quiere entender cómo funcionan las cosas, explicar los patrones que surgen de lo que se observa o predecir el comportamiento de un sistema en respuesta a algún cambio. Los sistemas reales a menudo son demasiado complejos o evolucionan muy lentamente para ser analizados mediante experimentos. 

## supervised learning

## Node

## Weight

## Bias

## softmax and softmax functio

## Layer

## Logistic Regression

## Gradient Descent

## Computation Graph

## multiclass classification

## one-hot vector

## pickling in Python

## cross-entropy loss

---

Preguntas generales

* [x] What is a model?
* [x] What is supervised learning?
* [x] What is a prediction?
* [x] What is a node?
* [x] What is a weight?
* [x] What is a bias?
* [x] What are activation functions?
* [x] Sigmoid?
* [x] Tanh?
* [x] Relu?
* [x] Softmax?
* [x] What is a layer?
* [x] What is a hidden layer?
* [x] What is Logistic Regression?
* [x] What is a loss function?
* [x] What is a cost function?
* [x] What is forward propagation?
* [x] What is Gradient Descent?
* [x] What is back propagation?
* [ ] What is a Computation Graph?
* [ ] How to initialize weights/biases
* [ ] The importance of vectorization
* [ ] How to split up your data
* [x] What is multiclass classification?
* [x] What is a one-hot vector?
* [ ] How to encode/decode one-hot vectors
* [ ] What is the softmax function and when do you use it?
* [x] What is cross-entropy loss?
* [x] What is pickling in Python?

----

Cosas por hacer:

* [ ] Contestar preguntas

``` 
 Neuronas sigmoides que simulan perceptrones, parte I 
Supongamos que tomamos todos los pesos y sesgos en una red de perceptrones y los multiplicamos por una constante positiva, c > 0. Muestre que el comportamiento de la red no cambia.
Neuronas sigmoides que simulan perceptrones, parte II 
Supongamos que tenemos la misma configuración que el último problema: una red de perceptrones. Suponga también que se ha elegido la entrada general a la red de perceptrones. No necesitaremos el valor de entrada real, solo necesitamos que la entrada se haya corregido. Suponga que los pesos y los sesgos son tales quew ⋅ x+ b ≠ 0 para la entrada Xa cualquier perceptrón particular de la red. Ahora reemplace todos los perceptrones en la red por neuronas sigmoides y multiplique los pesos y sesgos por una constante positivac > 0. Muestre que en el límite comoc → ∞el comportamiento de esta red de neuronas sigmoides es exactamente el mismo que el de la red de perceptrones. ¿Cómo puede fallar esto cuandow ⋅ x+ b = 0 para uno de los perceptrones?
```

* [ ] leer sobre redes neuronales recurrentes
* [ ] Hacer proyecto de [Una red simple para clasificar dígitos escritos a mano](http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons)
* [ ] Leer el libro completo de [neuralnetworksanddeeplearning](http://neuralnetworksanddeeplearning.com/chap1.html#sigmoid_neurons)
* [ ] Mirar que son derivadas parciales
* [ ] Ver sobre Renaissance hasta Two Sigma, 
* [ ] Mirar como calcular el error
* [ ] Tema SSE
* [ ] Tema de  función de pérdida 
* [ ] Mirar sobre retroalimentacion
* [ ] conversión de clases discretas 
* [ ] Ver sobre gradiantes 
* [ ] Codificación One-Hot
* [ ] Mirar el tema de los gradientes
* [ ] gradientes que desaparecen
* [ ] Mirar que tipo de funciones de perdida [aca](https://www.section.io/engineering-education/understanding-loss-functions-in-machine-learning/)
    -  Error absoluto medio (MAE)
    -  Error cuadrático medio (MSE)
    -  Error de sesgo medio (MBE)
    -  Error logarítmico cuadrático medio (MSLE)
    -  Pérdida de Huber
    -  Pérdida de entropía cruzada binaria
    -  Pérdida de entropía cruzada categórica
    -  Pérdida de bisagra
    -  Pérdida de divergencia de Kullback Leibler
* [ ]  ver el video de https://www.youtube.com/watch?v=tIeHLnjs5U8

Read after this http://colah.github.io/posts/2015-08-Backprop/

TEMAS POR VER:

why use For all matrix multiplications in the following tasks, please use numpy.matmul

* [ ] [Gradient descent]()
* [ ] [Calculus on Computational Graphs: Backpropagation]()
* [ ] [Backpropagation calculus]()
* [ ] [What is a Neural Network?]()
* [ ] [Supervised Learning with a Neural Network]()
* [ ] [Binary Classification]()
* [ ] [Logistic Regression]()
* [ ] [Logistic Regression Cost Function]()
* [ ] [Gradient Descent]()
* [ ] [Computation Graph]()
* [ ] [Logistic Regression Gradient Descent]()
* [ ] [Vectorization]()
* [ ] [Vectorizing Logistic Regression]()
* [ ] [Vectorizing Logistic Regression’s Gradient Computation]()
* [ ] [A Note on Python/Numpy Vectors]()
* [ ] [Neural Network Representations]()
* [ ] [Computing Neural Network Output]()
* [ ] [Vectorizing Across Multiple Examples]()
* [ ] [Gradient Descent For Neural Networks]()
* [ ] [Random Initialization]()
* [ ] [Deep L-Layer Neural Network]()
* [ ] [Train/Dev/Test Sets]()
* [ ] [Random Initialization For Neural Networks : A Thing Of The Past]()
* [ ] [Initialization of deep networks]()
* [ ] [Multiclass classification]()
* [ ] [Derivation: Derivatives for Common Neural Network Activation Functions]()
* [ ] [What is One Hot Encoding? Why And When do you have to use it?]()
* [ ] [Softmax function]()
* [ ] [What is the intuition behind SoftMax function?]()
* [ ] [Cross entropy]()
* [ ] [Loss Functions: Cross-Entropy]()
* [ ] [Softmax Regression (Note: I suggest watching this video at 1.5x - 2x speed)]()
* [ ] [Training Softmax Classifier (Note: I suggest watching this video at 1.5x - 2x speed)]()
* [ ] [numpy.zeros]()
* [ ] [numpy.random.randn]()
* [ ] [numpy.exp]()
* [ ] [numpy.log]()
* [ ] [numpy.sqrt]()
* [ ] [numpy.where]()
* [ ] [numpy.max]()
* [ ] [numpy.sum]()
* [ ] [numpy.argmax]()
* [ ] [What is Pickle in python?]()
* [ ] [pickle]()
* [ ] [pickle.dump]()
* [ ] [pickle.load]()
* [ ] see why the error in the 1 file
* [ ] [gradient](https://www.youtube.com/watch?v=z_xiwjEdAC4)

