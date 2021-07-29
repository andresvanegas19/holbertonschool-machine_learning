# Probability
![](https://i.imgur.com/y7WLeim.png)

> se refiere a las descripciones numéricas de la probabilidad de que ocurra un acontecimiento, o de la probabilidad de que una proposición sea verdadera. La probabilidad de un suceso es un número entre 0 y 1, donde, a grandes rasgos, 0 indica la imposibilidad del suceso y 1 la certeza.


Los objetivistas asignan números para describir un estado de cosas objetivo o físico. La versión más popular de la probabilidad objetiva es la probabilidad frecuentista, que afirma que la probabilidad de un evento aleatorio denota la frecuencia relativa de ocurrencia del resultado de un experimento cuando éste se repite indefinidamente. Esta interpretación considera que la probabilidad es la frecuencia relativa "a largo plazo" de los resultados. Una modificación de ésta es la probabilidad de propensión, que interpreta la probabilidad como la tendencia de algún experimento a producir un determinado resultado, aunque se realice una sola vez.

Los subjetivistas asignan números por probabilidad subjetiva, es decir, como un grado de creencia. El grado de creencia se ha interpretado como "el precio al que se compraría o vendería una apuesta que paga 1 unidad de utilidad si E, 0 si no E" La versión más popular de la probabilidad subjetiva es la probabilidad bayesiana, que incluye el conocimiento de los expertos así como los datos experimentales para producir probabilidades. El conocimiento de los expertos está representado por una distribución de probabilidad (subjetiva) a priori. Estos datos se incorporan a una función de probabilidad. El producto de la función a priori y la función de verosimilitud, cuando se normaliza, da lugar a una distribución de probabilidad posterior que incorpora toda la información conocida hasta la fecha. Por el teorema de la concordancia de Aumann, los agentes bayesianos cuyas creencias a priori son similares acabarán teniendo creencias posteriores similares. Sin embargo, unas creencias previas suficientemente diferentes pueden llevar a conclusiones diferentes, independientemente de la información que compartan los agentes

```
Cuanto mayor sea la probabilidad de un acontecimiento, más probable será que éste ocurra.
```

En la teoría de la probabilidad, el espacio muestral (también llamado espacio de descripción muestral o espacio de posibilidades) de un experimento o ensayo aleatorio es el conjunto de todos los resultados posibles de ese experimento.


Un concepto de probabilidad se extrae de la idea de resultados simétricos, es decir si hay N resultados simetricos, la probabilidad de que ocurra cualquiera de ellos se considera 1/N. Las probabilidades también se pueden pensar en términos de frecuencias relativas 

> Para algunos propósitos, es mejor pensar en la probabilidad como subjetiva.

### Probabilidad de un solo evento
Si conoce la probabilidad de que ocurra un evento, es fácil calcular la probabilidad de que el evento no ocurra. Si P (A) es la probabilidad del evento A, entonces 1 - P (A) es la probabilidad de que el evento no ocurra.

![](https://i.imgur.com/Wc2QvdG.png)

### Probabilidad de dos o más eventos independientes
Los eventos A y B son eventos independientes si la probabilidad de que ocurra el Evento B es la misma, ocurra o no el Evento A.

- Sucesos independientes:
     Los sucesos A y B son independientes si la probabilidad de que ocurra el suceso B es la misma independientemente de que ocurra o no el suceso A. Por ejemplo, si se lanzan dos dados, la probabilidad de que el segundo salga 1 es independiente de que el primer dado salga 1. Formalmente, esto se puede expresar en términos de
     ```
         conditional probabilities: P(A|B) = P(A) and P(B|A) = P(B).
     ```

## eventos independiente
### Probabilidad de A y B
La probabilidad de que ambos ocurran es el producto de las probabilidades de los eventos individuales. Más formalmente, si los eventos A y B son independientes, entonces la probabilidad de que ocurran tanto A como B es:
```
    P (A y B) = P (A) x P (B)
```
### Probabilidad de A o B
la probabilidad de que ocurra el evento A o el evento B es:
```
    P (A o B) = P (A) + P (B) - P (A y B)
```
- A ocurre y B no ocurre
- B ocurre y A no ocurre
- Tanto A como B ocurren



## La probabilidad condicional
La probabilidad de que ocurra el evento A dado que el evento B ya ocurrió se llama probabilidad condicional de A dado B. Simbólicamente, esto se escribe como P (A | B). La probabilidad de que llueva el lunes dado que llovió el domingo se escribiría como P (Lluvia el lunes | Lluvia el domingo). 

```
P (as en el segundo sorteo | un as en el primer sorteo)
```


La barra vertical "|" se lee como "dado", por lo que la expresión anterior es la abreviatura de: "La probabilidad de que se saque un as en el segundo sorteo dado que se sacó un as en el primer sorteo"

> Si los eventos A y B no son independientes, entonces P (A y B) = P (A) x P (B | A).


### Eventos independientes
Los eventos A y B son eventos independientes si la probabilidad de que ocurra el evento B es la misma ya sea que ocurra el evento A o no. Por ejemplo, si lanza dos dados, la probabilidad de que el segundo dado salga 1 es independiente de si el primer dado salió 1. Formalmente, esto puede expresarse en términos de probabilidades condicionales

Cuando dos eventos son independientes, la probabilidad de que ambos ocurran es el producto de las probabilidades de los eventos individuales. Más formalmente, si los eventos A y B son independientes, entonces la probabilidad de que ocurran tanto A como B es:

```
P (A y B) = P (A) x P (B)
```
```
P (A | B) = P ( A) y P (B | A) = P (B).
```

donde P (A y B) es la probabilidad de que ocurran los eventos A y B, P (A) es la probabilidad de que ocurra el evento A y P (B) es la probabilidad de que ocurra el evento B.

Si los eventos A y B son independientes, la probabilidad de que ocurra el evento A o el evento B es:

```
P (A o B) = P (A) + P (B) - P (A y B)
```

En esta discusión, cuando decimos "Ocurre A o B" incluimos tres posibilidades:

1. A ocurre y B no ocurre
2. B ocurre y A no ocurre
3. Tanto A como B ocurren


P(G|E) La probabilidad dada que encuentre una g entre el grupo E


### Disjoint Events
Disjoint events are events that never occur at the same time. These are also known as mutually exclusive events. 

These are often visually represented by a Venn diagram, such as the below. In this diagram, there is no overlap between event A and event B. These two events never occur together, so they are disjoint events.

| Features                             | Formula                               | Implement
| ------------------------------------ |:------------------------------------- |:--------
| ![](https://i.imgur.com/lde44eb.png) | The complement Ac. P(Ac) = 1- P(A)]   | things “complement” one another when they complete each other. A and Ac complete the whole sample space, as shown in the Venn diagram.
| ![](https://i.imgur.com/0OthtPN.png) | The intersection A ∩ B.               | the “intersection” is the place where things meet. A and B meet in A ∩ B, as shown in the Venn diagram.
| ![](https://i.imgur.com/bTztAdN.png) | The union A ∪ B. Note P(A ∪ B) = P(A) + P(B) – P(A ∩ B) | the “union” is what happens when things join together. A and B join together to make A ∪ B
| ![](https://i.imgur.com/epkyPqr.png) | Disjoint. Definition: A and B are disjoint when P(A ∩ B) =0. | things are “disjoint” when they’re not connected. Events A and B are disjoint when they can’t happen at the same time. In the Venn diagram, their areas are not connected. 
| No image  | Independent. Definition: A and B are independent when P(A ∩ B) = P(A)P(B). | things are “independent” when they don’t rely on each other. A and B are independent when knowing about one happening does not change how likely the other is. B happens P(B) of the time, so B also happens P(B) of the time that A happens – that is P(B) of P(A). So P(A ∩ B) = P(A)P(B).

![](https://i.imgur.com/4GBM3dg.png)

### What is independence? What is disjoint?
Disjoint events and independent events are different. Events are considered disjoint if they never occur at the same time; these are also known as mutually exclusive events. Events are considered independent if they are unrelated.

By independence of two events (Let event A and B) we mean that occurrence of A does not depend on the occurrence of B . By disjoint of two events we mean that whatever   that is happening in A can not happen in B. Let you are tossing a dice twice. In the first trial possible outcomes are either faces with numbers 1,2,3,4,5,6. Let us consider probability of coming a even number(A) and odd number(B). Now if the outcome is an even number (A) then it can not be an odd number(B) i.e. whatever happens in A can not happen in B. So A and B are disjoint. Now consider occurrence of no 6 in first(A) and second trial(B). Whether we get 6 on the first trial or not getting 6 in second trial does not depend upon that. i.e. we say B is independent of A.

Two events A and B are said to be disjoint if  𝐴∩𝐵=𝜙. 

Two events A and B are said to be independent if  𝑃(𝐴∩𝐵)=𝑃(𝐴)⋅𝑃(𝐵) 

Let us consider another example. This time we toss a fair dice one time. Let  𝐴={2,3},𝐵={3,4,5}. 

Then we see that  𝐴∩𝐵={3}  and  𝑃(𝐴∩𝐵)=𝑃(𝐴)⋅𝑃(𝐵)  which imply A and B are independent events.

![](https://i.imgur.com/ku0W0MX.png)

![](https://i.imgur.com/FXfiDJ2.png)

an information of one is wont change the another probability
toda la poblacion que esta en la derecha se calcula esa probabilidad para saber cual es

## Addition Rule in Probability
If A and B are two events in a probability experiment, then the probability that either one of the events will occur is:
```
P(A or B) = P(A) + P(B) − P(A and B)
P(A ∪ B) = P(A) + P(B) − P(A ∩ B)
```
## GENERAL RULE FOR UNION (OR)
## Mutually Exclusive Events
```
 P(A or B) = P(A) + P(B) - P(A and B)
```
When A and B are disjoint P(A and B) = 0 and the rule reverts to the addition rule of before
Venn diagrams are a help in finding probabilities for unions, because you can just think of adding and subtracting areas.
## CONDITIONAL PROBABILITY
is the probability assigned to an event when we KNOW some other event has ALREADY occurred.
P(A|B) is the notation for conditional probability "read" probability of A given that B has occurred or shorthand probability of A given B. 
The formula for  P(A|B) = P(A and B) / P(B) where P(B) > 0 which can be manipulated to generate.

## GENERAL MULTIPLICATION RULE FOR ANY TWO EVENTS
```
P(A and B) = P(A) x P(B|A)
```
The UNION of a collection of events is the event that ANY of them occur.
The INTERSECTION of any collection of events is the event that ALL of them occur.

To extend the multiplication rule to the probability that all of several events occur, the key is to condition each event on the occurrence of all of the preceding events, like stacking....for events A, B, and C this looks like
P(A and B and C) = P(A)P(B|A) P(C|A and B)


# Permutaciones y combinaciones
## Pedidos posibles
Es cuando quieres saber cuantas combinaciones puede haber en un suceso X

## Regla de multiplicación
Cunado quieres saber cuantas psobilidades hay entre escoger varias opciones, se debe multiplicar estas opciones para percibir este numero

## Permutaciones
P es el número de permutaciones de n cosas tomadas r a la vez,  es el número de formas en que se pueden seleccionar r cosas de un grupo de n cosas

![](https://i.imgur.com/3WnAfEY.png)

las permutaciones se refieren al número de formas de elegir más que al número de resultados posibles. Cuando no se considera el orden de elección, se utiliza la fórmula para combinaciones.

## Combinaciones
- con cuántas combinaciones diferentes de dos piezas podrías terminar
 n C r es el número de combinaciones para n cosas tomadas r a la vez.

![](https://i.imgur.com/QEqMhKT.png)

# Probability distribution
**Una distribución de probabilidad es una descripción matemática de las probabilidades de eventos**

![](https://i.imgur.com/fJMAM1U.png)

Una distribución de probabilidad es la función matemática que da las probabilidades de ocurrencia de diferentes resultados posibles para un experimento. Es una descripción matemática de un fenómeno aleatorio en términos de su espacio muestral y las probabilidades de eventos (subconjuntos del espacio muestral).

Por ejemplo, si X se usa para denotar el resultado de un lanzamiento de moneda ("el experimento"), entonces la distribución de probabilidad de X tomaría el valor 0.5 para X  = cara y 0.5 para X  = cruz (asumiendo que la moneda es justo). Los ejemplos de fenómenos aleatorios incluyen las condiciones climáticas en una fecha futura, la altura de una persona seleccionada al azar, la fracción de estudiantes varones en una escuela, los resultados de una encuesta que se realizará

--------------------------------------
# Glosary

### Resultado favorable
Un resultado favorable es el resultado de interés. El término "resultado favorable" no significa necesariamente que el resultado sea deseable: en algunos experimentos, el resultado favorable podría ser el fracaso de una prueba o la aparición de un evento indeseable.


### Probabilidad condicional

La probabilidad de que ocurra el suceso A dado que ya ha ocurrido el suceso B se llama probabilidad condicional de A dado B. Simbólicamente se escribe como P(A|B). La probabilidad de que llueva el lunes dado que llovió el domingo se escribiría como P(Lluvia el lunes | Lluvia el domingo).

### Probability theory
una rama de las matemáticas que se ocupa del análisis de fenómenos aleatorios. El resultado de un evento aleatorio no se puede determinar antes de que ocurra, pero puede ser cualquiera de varios resultados posibles. Se considera que el resultado real está determinado por casualidad.

### Inference
The act of drawing conclusions about a population from a sample.

### Inferential Statistics
Rama de la estadística que se ocupa de sacar conclusiones sobre una población a partir de una muestra. Esto generalmente se hace a través de un muestreo aleatorio, seguido de inferencias sobre la tendencia central o cualquiera de los otros aspectos de una distribución.

### Population
Una población es el conjunto completo de observaciones que le interesan a un investigador. Compare esto con una muestra que es un subconjunto de una población. Una población se puede definir de una manera conveniente para un investigador. Las estadísticas inferenciales se calculan a partir de datos de muestra para hacer inferencias sobre la población .

### sample
A sample is a subset of a population, often taken for the purpose of statistical inference. Generally, one uses a random sample. 

### Distribution
#### Frequency Distribution
La distribución de datos empíricos se llama distribución de frecuencia y consiste en un recuento del número de ocurrencias de cada valor. Si los datos son continuos, se utiliza una distribución de frecuencia agrupada . Normalmente, una distribución se representa mediante un polígono de frecuencias o un histograma .

Las ecuaciones matemáticas se utilizan a menudo para definir distribuciones. La distribución normal es, quizás, el ejemplo más conocido. Muchas distribuciones empíricas se aproximan bien mediante distribuciones matemáticas como la distribución normal.


### Resultado favorable
Un resultado favorable es el resultado de interés. Por ejemplo, se podría definir un resultado favorable en el lanzamiento de una moneda como cara. El término "resultado favorable" no significa necesariamente que el resultado sea deseable; en algunos experimentos, el resultado favorable podría ser el fracaso de una prueba o la ocurrencia de un evento indeseable.

### Mutually Exclusive Events
If two events have no elements in common (Their intersection is the empty set.), the events are called mutually exclusive.   Thus, P(A∩B)=0 .  This means that the probability of event A and event B happening is zero.  They cannot both happen.

### Variable aleatoria
una variable aleatoria , una cantidad aleatoria , una variable aleatoria o una variable estocástica se describe informalmente como una variable cuyos valores dependen de los resultados de un fenómeno aleatorio. El tratamiento matemático formal de las variables aleatorias es un tema en la teoría de la probabilidad . En ese contexto, una variable aleatoria se entiende como una función medible definida en un espacio de probabilidad que mapea desde el espacio muestral a los números reales


# notation

- P(A) refers to the probability that event A will occur.
- P(A|B) refers to the conditional probability that event A occurs, given that event B has occurred.
- P(A') refers to the probability of the complement of event A.
- ==P(A ∩ B)== refers to the probability of the intersection of events A and B.
- P(A ∪ B) refers to the probability of the union of events A and B.
- E(X) refers to the expected value of random variable X.
- b(x; n, P) refers to binomial probability.
- b*(x; n, P) refers to negative binomial probability.
- g(x; P) refers to geometric probability.
- h(x; N, n, k) refers to hypergeometric probability.
- El espacio muestral, a menudo denotado por Omega 

![](https://i.imgur.com/AgqjfSz.png)

### questions
- [ ] What is probability?
- [ ] Basic probability notation
- [ ] What is independence? What is disjoint?
- [ ] What is a union? intersection?
- [ ] What are the general addition and multiplication rules?
- [ ] What is a probability distribution?
- [ ] What is a probability distribution function? probability mass function?
- [ ] What is a cumulative distribution function?
- [ ] What is a percentile?
- [ ] What is mean, standard deviation, and variance?
- [ ] Common probability distributions
- [ ] Que es la probabilidad bayesiana
- [ ] En que momento se usan las variables aletorias
- [ ] Como se usan las vairbales aleatorias
- [ ] Que es una funcion de masa de probabilidad
- [ ] Diferencia entre discreta y continuas
- [ ] Como funciona la funcion de densidad de probabilidad
- [ ] Cual es la probabilidad infinitesimal



use various irrational numbers and functions. 
![](https://i.imgur.com/sVHCU9H.png)
