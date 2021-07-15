# Algebra lineal

Linear Algebra is a branch of mathematics that concisely describes the coordinates and interactions of planes in higher dimensions and perform operations on them.

Think of it as an extension of algebra (dealing with unknowns) into an arbitrary number of dimensions. Linear Algebra is about working on linear systems of equations (linear regression is an example: y = Ax). Rather than working with scalars, we start working with vectors and matrices.

Se orienta a la generalización de las operaciones aritméticas a través de signos, letras y números. En el álgebra, las letras y los signos representan a otra entidad a través de un simbolismo.

Se conoce como álgebra lineal a la especialización del álgebra que trabaja con matrices, vectores, espacios vectoriales y ecuaciones de tipo lineal. 

Las preguntas que se hacen en cada tema son:

* Que es?
* porque se usa?
* Cuando se usa?
* De donde salio la opercaicon?

****

# - What is a **vector**? <br>

***Un vector es una herramienta matemática que nos permite representar magnitudes en las que no sólo importa la intensidad (o módulo), sino también la dirección y el sentido en la que están aplicadas.***
<br>
un vector tiene entradas. Es cualquier conjunto de cosa que se pueden sumar entre ellas y se puede multiplicar por un número.
<br>
Entre la suma de vectores es un movimiento en el espacio
<br>
Alargar el vector y multiplicar por vectores se le llama escalar, donde se alargan los vectores y se escogen. Cada coordenada es como escalar. para la multiplicación de vector hay dos formas, producto punto o producto escalar, que es un escalar o producto cruz que termina siendo un vector. Otra operación es una compuesta.
<br>
Si se esta trabajando con un conjunto de vectores es recomendable trabajarlos como puntos, es asi que sera mas facil diferenciarlos en un plano y trabajar con ellos. La base en los vectores se reficere a un conjunto de vectores linealmente independientes que genera todo el espacio

  + ## Componentes

		Las componentes son los valores reales para cada eje del sistema de coordenadas. En el plano un vector tiene dos componentes, generalmente x e y, en el espacio necesitamos tres componentes. Los componentes se puede inferenciar que son escalares ya que pueden alargar o encoger vectores

  + ## Módulo o Norma

		El módulo o norma de un vector nos habla del tamaño del vector, la magnitud o intensidad que tiene, en otras palabras es cuánto mide el vector desde el origen hasta el punto final. <br>

    - ## real coordinate

        all posible real values - is a coordinate space over the real numbers. This means that it is the set of the n-tuples of real numbers (sequences of n real numbers). With component-wise addition and scalar multiplication, it is a real vector space.

Esto es un vector -> v = -1i + 2i<br>
los vectores unitarios contienen una magnitud de 1, para allar el vector unicario, se divide el vector por su magnitud <br>

Una combinación lineal de dos o más vectores es el vector que se obtiene al sumar esos vectores multiplicados por algunos escalares. Es decir, una combinación lineal es una expresión de la forma:

![](https://i.imgur.com/V7WwJRL.png)
![](https://i.imgur.com/y6I5Yk3.png)

### Span

 the linear span (also called the linear hull or just span) of a set S of vectors (from a vector space), denoted span(S), [1] is the smallest linear subspace that contains the set. It can be characterized either as the intersection of all linear subspaces that contain S, or as the set of linear combinations of elements of S. The linear span of a set of vectors is therefore a vector space. Spans can be generalized to matroids and modules.
 
Crear una combinación lineal de vectores es muy simple. Dado un conjunto de vectores, como (v₁, v₂, v₃).<br>
Una combinación lineal es el vector que se obtiene al sumar un múltiplo de v₁, un múltiplo de v₂ y un múltiplo de v₃. 

``` 

                    av + bw
            where a and b are scalars
```

Los múltiplos exactos pueden ser cualquier número que queramos. Los vectores v₁, v₂ y v₃ son como pintura roja, amarilla y azul, y una combinación lineal es el acto de mezclar esos vectores para crear un nuevo vector. <br>
Cada una de estas combinaciones lineales, por sí solas, puede considerarse como c₁v₁ + c₂v₂ + c₃v₃ donde cada c es un número real. El conjunto de todas estas combinaciones lineales se denomina intervalo de (v₁, v₂, v₃) y, a veces, se escribe simplemente como intervalo (v₁, v₂, v₃).

![](https://i.imgur.com/vwPQF6J.png)

Si vectores son dependientes, el lapso es el mismo que si quitamos uno de los vectores. Si los vectores son independientes, el intervalo cambia si elimina un vector. Los vectores dependientes son como tener rojo, amarillo y naranja, mientras que los vectores independientes son como tener rojo y amarillo. El lapso, la cantidad total de colores que podemos hacer, es el mismo para ambos.

### review linear independet/dependet

    - Given the equation c₁v₁ + c₂v₂ + … + cnvn = 0v, where all the v’s are vectors, 0v is the zero vector, and the c’s are scalars, then setting all the c’s to zero is called the trivial solution.
    - A set of vectors is linearly dependent if there exists a non-trivial solution to the equation c₁v₁ + c₂v₂ + … + cnvn = 0v (which actually implies infinite solutions).
    - A set of vectors are linearly independent if there doesn’t exist non-trivial solutions to the equation c₁v₁ + c₂v₂ + … + cnvn = 0v
    - If you put a set of vectors into a matrix, the vectors are independent if and only if every column is a pivot column.
    - Another way of saying the above point: if you put your set of matrix and solve the system of linear equations, dependent vectors will have infinite solutions, but independent vectors will only have one solution (that sneaky trivial solution)
    - A set of vectors is dependent if there are more columns than rows when you put them into a matrix.
    - Another way of saying the above point: if there are more vectors than the dimension of your space, they’re dependent. For example, if your vectors are in 2-D space, two vectors may or may not be dependent, but three vectors or four vectors will always be dependent.

****

# - What is the dot product? <br>

![image alt](https://github.com/andresvanegas19/holbertonschool-machine_learning/blob/main/math/0x00-linear_algebra/img/baseline.png)

<br>
Is a measure of how similar two vectors are <br> `𝐀⋅𝐁=‖𝐀‖‖𝐁‖cos𝜃,where 𝜃 is the angle between 𝐴 and 𝐵.`

![](https://www.youtube.com/watch?v=BcxfxvYCL1g&t=14s)
Ax . Bx + Ay . By --- A = (1, 2) --- B = (4, 5)
The dot product can be used for computing the angle 𝛼 between two vectors 𝑎 and 𝑏:
The dot product represents the actual "multiplication overlap" between the vectors. On its own it's just a number, and we often need to compare it to the max possible overlap.

  + Add vectors: Accumulate the growth 

contained in several vectors.

  + Multiply by a constant: Make an existing vector stronger (in the same direction).
  + Dot product: Apply the directional growth of one vector to another. The result is how much stronger we've made the original vector (positive, negative, or zero).

``` Text
 3 x 4 como un producto escalar
 (3, 0) . (4, 0)
 El número 3 es "crecimiento direccional" en una sola dimensión,
 y 4 es "crecimiento direccional" en esa misma dirección.
 3 x 4 = 12 significa que obtenemos un crecimiento de 12x en una sola dimensión. 
```

Dot products are very geometrical objects. They actually encode relative information about vectors, specifically they tell us "how much" one vector is in the direction of another. Particularly, the dot product can tell us if two vectors are (anti)parallel or if they are perpendicular.

We have the formula 𝑎⃗ ⋅𝑏⃗ =‖𝑎⃗ ‖‖𝑏⃗ ‖cos(𝜃), where 𝜃 is the angle between the two vectors in the plane that they make. If they are perpendicular, 𝜃=90∘, 270∘ so that cos(𝜃)=0. This tells us that the dot product is zero. This reasoning works in the opposite direction: if the dot product is zero, the vectors are perpendicular.

This gives us a quick way to tell if two vectors are perpendicular. It also gives easy ways to do projections and the like.

****

# - Linear subspace

linear subspace, also known as a vector subspace is a vector space that is a subset of some larger vector space. A linear subspace is usually simply called a subspace when the context serves to distinguish it from other types of subspaces.

### Vector Spaces

Vector space, a set of multidimensional quantities, known as vectors, together with a set of one-dimensional quantities, known as scalars, such that vectors can be added together and vectors can be multiplied by scalars while preserving the ordinary arithmetic properties (associativity, commutativity, distributivity, and so forth).
![](https://i.imgur.com/CcnDeNi.png)

### Subspace

a set A is a subset of a set B if all elements of A are also elements of B; B is then a superset of A. It is possible for A and B to be equal; if they are unequal, then A is a proper subset of B. The relationship of one set being a subset of another is called inclusion (or sometimes containment). A is a subset of B may also be expressed as B includes (or contains) A or A is included (or contained) in B.

The subset relation defines a partial order on sets. In fact, the subsets of a given set form a Boolean algebra under the subset relation, in which the join and meet are given by intersection and union, and the subset relation itself is the Boolean inclusion relation.

### closed under operation

A set is closed under a particular operation if, when you apply the operation to two members of the set, the result is also a member of the set. So the set of integers is closed under multiplication, because if you multiply two integers the answer is an integer. However, it is not closed under division because you could get an answer that is not an integer

### set

A set is a collection of things (usually numbers)

### basis

A linearly independent spanning set for V is called a basis, 
Any vector space V has a basis. All bases for V are of the same cardinality.

a set B of vectors in a vector space V is called a basis if every element of V may be written in a unique way as a finite linear combination of elements of B. The coefficients of this linear combination are referred to as components or coordinates of the vector with respect to B. The elements of a basis are called basis vectors.

Equivalently, a set B is a basis if its elements are linearly independent and every element of V is a linear combination of elements of B.[1] In other words, a basis is a linearly independent spanning set.

A vector space can have several bases; however all the bases have the same number of elements, called the dimension of the vector space.

This article deals mainly with finite-dimensional vector spaces. However, many of the principles are also valid for infinite-dimensional vector spaces.

![](https://i.imgur.com/bLnyU3x.png)

A basis B of a vector space V over a field F (such as the real numbers R or the complex numbers C) is a linearly independent subset of V that spans V. This means that a subset B of V is a basis if it satisfies the two following conditions:

![](https://i.imgur.com/I3AHXPN.png)

### closure

 is when an operation (such as "adding")
on members of a set (such as "real numbers")
always makes a member of the same set.

 a set is closed under an operation if performing that operation on members of the set always produces a member of that set. For example, the positive integers are closed under addition, but not under subtraction: 1 − 2 is not a positive integer even though both 1 and 2 are positive integers. Another example is the set containing only zero, which is closed under addition, subtraction and multiplication (because 0 + 0 = 0, 0 − 0 = 0, and 0 × 0 = 0).

Similarly, a set is said to be closed under a collection of operations if it is closed under each of the operations individually.

![](https://i.imgur.com/x99UnSX.png)

### null space

The null space of any matrix A consists of all the vectors B such that AB = 0 and B is not zero. It can also be thought as the solution obtained from AB = 0 where A is known matrix of size m x n and B is matrix to be found of size n x k. The size of the null space of the matrix provides us with the number of linear relations among attributes.

 es otro espacio fundamental en una matriz, siendo el conjunto de todos los vectores que terminan en cero cuando se les aplica la transformación.

01. Ab = 0 implies every row of A when multiplied by B goes to zero.
2. Variable values in each sample(represented by a row) behave the same.
03. This helps in identifying the linear relationships in the attributes.
04. Every null space vector corresponds to one linear relationship.

### Nullity

Nullity can be defined as the number of vectors present in the null space of a given matrix. In other words, the dimension of the null space of the matrix A is called the nullity of A. The number of linear relations among the attributes is given by the size of the null space. The null space vectors B can be used to identify these linear relationship.

### Rank Nullity Theorem

The rank-nullity theorem helps us to relate the nullity of the data matrix to the rank and the number of attributes in the data. The rank-nullity theorem is given by

``` 

Nullity of A + Rank of A = Total number of attributes of A (i.e. total number of columns in A)
```

### Rank

Rank of a matrix refers to the number of linearly independent rows or columns of the matrix.
![](https://i.imgur.com/5gr5xZe.png)

****

# - What is a matrix?<br>

![image alt](https://github.com/andresvanegas19/holbertonschool-machine_learning/blob/main/math/0x00-linear_algebra/img/matrix.png)

<br>
Una matriz es una forma compacta pero general de representar cualquier transformación lineal.<br>
Entonces, una transformación lineal se puede representar mediante una matriz de coeficientes. El tamaño de la matriz le dice el número de dimensión del dominio y los espacios de la imagen. La composición de dos transformadas lineales corresponde al producto de sus matrices. La inversa de una transformada lineal corresponde a la inversa de la matriz.<br>
The numbers of rows and columns of a matrix are called its dimensions.<br>
Un determinante mide el volumen de la imagen de un cubo unitario mediante la transformación; es un solo número. Los determinantes son una herramienta fundamental en la resolución de sistemas de ecuaciones lineales. una transformación lineal se puede descomponer en una rotación pura, una escala pura (anisotrópica) y otra rotación pura. Solo la escala deforma los volúmenes y el determinante de la transformada es el producto de los coeficientes de escala. is representing linear transformations. <br>
The relationship between matrices and linear transformations comes from the fact that a linear transformation is completely specified by the values it takes on a basis for its domain. <br>
**Las matrices nos proporcionan un lenguaje para describir estas transformaciones donde las columnas representan esas coordenadas y la multiplicacion de un vector por una matriz es la forma de calcular  lo que esa transformacion hace con el vector**

``` 

An 𝑚 by 𝑛 matrix is a function of two variables, the first of which has domain {1,2,…,𝑚} and the second of which has domain {1,2,…,𝑛}.
```

``` 

Una matrix es una determinada transformacion del espacio, cada numero que tiene una matriz es un elemento
```

### What is the Inverse of a Matrix?

The Inverse of a Matrix is the same idea but we write it A-1

![](https://i.imgur.com/YfoWfsO.png)

When we multiply a matrix by its inverse we get the Identity Matrix (which is like "1" for matrices):

![](https://i.imgur.com/9LchHyZ.png)

Because with matrices we don't divide! Seriously, there is no concept of dividing by a matrix. But we can multiply by an inverse, which achieves the same thing.

![](https://i.imgur.com/j0XHfNg.png)

Muchas operaciones de álgebra matricial / lineal pueden considerarse como composiciones de transformaciones de un punto en el espacio. Transformar por la matriz te lleva en una dirección, transformar por la inversa de la matriz te lleva a la dirección opuesta.

### Algorithm(Row Reduction)

Every matrix is row equivalent to one and only one matrix in reduced row echelon form.
![](https://i.imgur.com/wRCHHGK.png)

To perform Gauss-Jordan Elimination:

01. Swap the rows so that all rows with all zero entries are on the bottom

03. Swap the rows so that the row with the largest, leftmost nonzero entry is on top.

05. Multiply the top row by a scalar so that top row's leading entry becomes 1.

07. Add/subtract multiples of the top row to the other rows so that all other entries in the column containing the top row's leading entry are all zero.

09. Repeat steps 2-4 for the next leftmost nonzero entry until all the leading entries are 1.

11. Swap the rows so that the leading entry of each nonzero row is to the right of the leading entry of the row above it.

Gaussian Elimination helps to put a matrix in row echelon form, while Gauss-Jordan Elimination puts a matrix in reduced row echelon form. For small systems (or by hand), it is usually more convenient to use Gauss-Jordan elimination and explicitly solve for each variable represented in the matrix system. However, Gaussian elimination in itself is occasionally computationally more efficient for computers. Also, Gaussian elimination is all you need to determine the rank of a matrix (an important property of each matrix) while going through the trouble to put a matrix in reduced row echelon form is not worth it to only solve for the matrix's rank.

---

# - What is a transpose? <br>

![image alt] (https://github.com/andresvanegas19/holbertonschool-machine_learning/blob/main/math/0x00-linear_algebra/img/Matrix_transpose.gif)<br>
The transpose of a matrix is a new matrix whose rows are the columns of the original. <br>
Transposición de vectores es fundamental para proporcionar las propiedades de tamaños y ángulos. De hecho, la razón por la que el álgebra lineal es tan útil es que los vectores son los objetos matemáticos más simples para los que se pueden proporcionar nociones de tamaños y ángulos y, por lo tanto , de similitud . <br>
Cambiar la dimensionalidad se vuelve importante para facilitar el análisis. Un conjunto de datos en mi ejemplo tiene una fila que representa las lecturas de todos los sensores en el tiempo t, mientras que el otro representa la lectura de un sensor durante el tiempo t1, t2, t3, etc. <br>
to exchange places, after this operation the new matrix so obtained has its rows as the columns and columns as the rows of the old matrix 

* What is the shape of a matrix? <br>

The shape property is usually used to get the current shape of an array, but may also be used to reshape the array in-place by assigning a tuple of array dimensions to it. As with numpy.reshape, one of the new shape dimensions can be -1, in which case its value is inferred from the size of the array and the remaining dimensions. Reshaping an array in-place will fail if a copy is required.
**returns a tuple with each index having the number of corresponding elements.returns a tuple with each index having the number of corresponding elements.**

* What is an axis? <br>

Series object has only “axis 0” because it has only one dimension.
An axis is a line with respect to which a curve or figure is drawn, measured, rotated, etc. The most common axes encountered are commonly the mutually perpendicular Cartesian axes in the plane or in space.
A DataFrame object has two axes: “axis 0” and “axis 1”. “axis 0” represents rows and “axis 1” represents columns. Now it’s clear that Series and DataFrame share the same direction for “axis 0” – it goes along rows direction.

* What are element-wise operations / Hadamard produc? <br>

Is a binary operation that takes two matrices of the same dimensions and produces another matrix of the same dimension as the operands, where each element i, j is the product of elements i, j of the original two matrices. It is to be distinguished from the more common matrix product. <br>
An element-wise operation allows you to distribute the operation over the elements of a data container.  The syntax for this is to use a tilde (~) after the given operator or function name. <br>
An element-wise function allows you to apply a function to the elements of a data container.  As with any function call, you must use parentheses. <br>
Pandas is designed for operating vector wise operations i.e. taking entire column and operate some function. This you can term as column wise operation. But in some cases you may need to operate element by element (i.e. element wise operation). This type operation is not very efficient.

* How do you concatenate vectors/matrices? <br>
  + [append a vector to a matrix](https://stackoverflow.com/questions/20978757/how-to-append-a-vector-to-a-matrix-in-python)
  + [Concatenate matrix with another matrix](https://stackoverflow.com/questions/33405219/concatenating-numpy-vector-and-matrix-horizontally)
  + [More concatening](https://cmdlinetips.com/2018/04/how-to-concatenate-arrays-in-numpy/)

* What is matrix multiplication? <br>

![image alt](https://github.com/andresvanegas19/holbertonschool-machine_learning/blob/main/math/0x00-linear_algebra/img/hand_ani.gif)

<br>
In mathematics, particularly in linear algebra, matrix multiplication is a binary operation that produces a matrix from two matrices. For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix. The resulting matrix, known as the matrix product, has the number of rows of the first and the number of columns of the second matrix. The product of matrices {\displaystyle A}A and {\displaystyle B}B is then denoted simply as {\displaystyle AB}AB

![image alt](https://github.com/andresvanegas19/holbertonschool-machine_learning/blob/main/math/0x00-linear_algebra/img/rotacion.png)

<br>

* Las herramientas que se usan en Numpy y que es esto?
* What is parallelization and why is it important?

* What is broadcasting? <br>

The term broadcasting refers to how numpy treats arrays with different Dimension during arithmetic operations which lead to certain constraints, the smaller array is broadcast across the larger array so that they have compatible shapes.
Broadcasting provides a means of vectorizing array operations so that looping occurs in C instead of Python as we know that Numpy implemented in C. It does this without making needless copies of data and which leads to efficient algorithm implementations. There are cases where broadcasting is a bad idea because it leads to inefficient use of memory that slow down the computation. <br>
**Broadcasting Rules:** <br>

Broadcasting two arrays together follow these rules:<br>

* If the arrays don't have the same rank then prepend the shape of the lower rank array with 1s until both shapes have the same length.
* The two arrays are compatible in a dimension if they have the same size in the dimension or if one of the arrays has size 1 in that dimension.
* The arrays can be broadcast together iff they are compatible with all dimensions.
* After broadcasting, each array behaves as if it had shape equal to the element-wise maximum of shapes of the two input arrays.
* In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension.

Well, the meaning of trailing axes is explained on the linked documentation page. If you have two arrays with different dimensions number, say one 1x2x3 and other 2x3, then you compare only the trailing common dimensions, in this case 2x3. But if both your arrays are two-dimensional, then their corresponding sizes have to be either equal or one of them has to be 1. Dimensions along which the array has size 1 are called singular, and the array can be broadcasted along them.

Broadcasting is possible if the following rules are satisfied:

  + Array with smaller ndim than the other is prepended with '1' in its shape.
  + Size in each dimension of the output shape is maximum of the input sizes in that dimension.
  + An input can be used in calculation, if its size in a particular dimension matches the output size or its value is exactly 1.
  + If an input has a dimension size of 1, the first data entry in that dimension is used for all calculations along that dimension.

In your case you have a 2x2 and 4x2 and 4 != 2 and neither 4 or 2 equals 1, so this doesn't work.

---

# More topics

### System of linear equations

A System of Linear Equations is when we have two or more linear equations working together.

![](https://i.imgur.com/Ec8H9FX.png)

Only simple variables are allowed in linear equations, A Linear Equation can be in 2 dimensions
"Independent" means that each equation gives new information. Otherwise they are "Dependent". Also called "Linear Independence" and "Linear Dependence"

![](https://i.imgur.com/RI1boJe.png)

When there is no solution the equations are called "inconsistent". One or infinitely many solutions are called "consistent"

### sistema de ecuaciones homogeneo <br>

Si un sistema de m ecuaciones y n incógnitas tiene todos los términos independientes nulos se dice que es homogéneo.

### Triviality <br>

 the adjective trivial is often used to refer to a claim or a case which can be readily obtained from context, or an object which possesses a simple structure, The noun triviality usually refers to a simple technical aspect of some proof or definition.

### Transformacion lineal  <br>

`

``` text
A linear transformation is a function 𝑓 of vectors which has the following properties:
𝑓(𝑥+𝑦)=𝑓(𝑥)+𝑓(𝑦) for any vectors 𝑥 and 𝑦.
𝑓(𝑎𝑥)=𝑎𝑓(𝑥) for any vector 𝑥 and any scalar 𝑎.
````

These properties are what it takes to ensure that the function 𝑓 has "no curvature". So it's like a straight line, but possibly in higher dimensions.

One essential way to understand matrices is to consider them as a collection of column vectors.

Por ser función, tiene su dominio y su codominio, con la particularidad de que éstos son espacios vectoriales. Las siguientes reglas se deben cumplir para que sea una transformacion lineal.

  + Las rectas siguien siendo rectas 
  + No se pueden curvar
  + El origen se mantiene fijo

Son una forma de transformar el espacio de manera que las lineas de la cuadricula permanezcan paralelas y equidistantes de manera que el origen permanezcan fijo

### Dominio. <br>

Para las funciones de varias variables el dominio es una regla de asignación de dimensiones, variación conferida a la cantidad de variables independientes, y los números reales.

### Funcion <br>

una función es una relación entre dos variables, una independiente (x) y otra dependiente (y) y por cada valor de x le corresponde UN ÚNICO VALOR DE y. A cada valor de x le corresponde UN ÚNICO VALOR DE y

### Escalar <br>

La magnitud escalar es la cantidad que podemos medir de una cierta propiedad que no depende de su dirección o posición en el espacio. Un escalar es una cantidad que tiene una magnitud pero no dirección. La magnitud escalar se refiere a la medida como tal.

### Aplicar vectores <br>

¿Cuánta energía / empuje le da un vector al otro?

### Normalizar un vector <br>

Normalizar, consite en tomar a un vector  distinto de cero, y con él obtener un vector , de la misma dirección y sentido que  pero con magnitud uno. En ciertas ocasiones, a los vectores unitarios también se les da el nombre de vector normalizado. Este tipo de vectores se usan de manera reiterada en caso de trabajar con derivadas, por lo tanto son relevantes en el caso de la física. Por ello, es menester conocerlos. Para calcularlos, hay que hallar la derivada de la curva en un punto dado.

# LINEAR ALGEBRA, BASIC NOTIONS

* E -> member
* ![](https://i.imgur.com/DzkWvGP.png)

https://en.wikibooks.org/wiki/Linear_Algebra/Notation

NOTATIONS <br>
http://linear.ups.edu/html/notation.html

![](https://i.imgur.com/NCa5Yzk.png)
![](https://i.imgur.com/hKakpAN.png)

****

# preguntas

que es plot
que es catter plot line grap bar graph histogram que es matplotlib how to label a plot how to scale an axis que es arange en numpy en que momento usar un grafico como funciona el subplot ejmplos y ejercicios como escalar un grafico para que es el log yporque se usa en el grafico buenas practicas para usar matplotlib que es un random seed y en que momento usarlos que son los bins de los histogramas como hacer el grid en python algunos ejercicios como sirve el dandom randient mirar cuales son los ticks mirar cada parte que es una gadiante en em momento usar scatter para que es el alpha Que es PCA y tupos de color map
Matrix Factorization
