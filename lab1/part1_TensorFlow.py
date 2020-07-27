### Intro to TensorFlow ###

import tensorflow as tf 
import mitdeeplearning as mdl
import numpy as np 
import matplotlib.pyplot as plt

###
#### 1.1 N-D Tensors ####
###

# 0-d Tensors
sport = tf.constant("Tennis", tf.string)
number = tf.constant(1.41421356237, tf.float64)

print("'sport' is a {}-d Tensor".format(tf.rank(sport).numpy()))
print("'sport' is a {}-d Tensor".format(tf.rank(number).numpy()))

# 1-d Tensors
sports = tf.constant(["Tennis", "Basketball"], tf.string)
numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)

print("'sports' is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
print("'numbers' is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))

# Defining higher-order Tensors

# 2-d Tensors
matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]], tf.float16)
assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(matrix).numpy() == 2

row_vector = matrix[1]
column_vector = matrix[:,1]
scalar = matrix[1,1]

print("`row_vector`: {}".format(row_vector.numpy()))
print("`column_vector`: {}".format(column_vector.numpy()))
print("`scalar`: {}".format(scalar.numpy()))


# 4-d Tensors
images = tf.zeros([10, 256, 256, 3], tf.int32)
assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
print("images is a {}-d Tensor with shape: {}".format(tf.rank(images).numpy(), tf.shape(images)))


###
#### 1.2 Computations on Tensors ####
###

a = tf.constant(15)
b = tf.constant(61)

c1 = tf.add(a, b)
c2 = a + b      # No diffrences
print(c1)
print(c2)

def func(a, b):
    c = tf.add(a, b)
    d = tf.subtract(b, 1)
    e = tf.multiply(c, d)
    return e

a, b = 1.5, 2.5
print(func(a, b))


print('\n')


###
#### 1.3 Neural networks in TensorFlow ####
###

class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer, self).__init__()
        self.n_output_nodes = n_output_nodes
    
    def build(self, input_shape):
        d = int(input_shape[-1])
        
        # Random weights
        self.W = self.add_weight("weight", shape = [d, self.n_output_nodes])
        self.b = self.add_weight("bias", shape = [1, self.n_output_nodes])

    def call(self, x):
        m = tf.matmul(x, self.W)
        z = tf.add(m, self.b)
        y = tf.sigmoid(z)
        return y


tf.random.set_seed(1)
layer = OurDenseLayer(3)
layer.build((1, 2))
x_input = tf.constant([[1, 2.]], shape = (1, 2))
y = layer.call(x_input)
print(y.numpy())    # Test Output
mdl.lab1.test_custom_dense_layer_output(y)

print('\n')

### Defining a neural network using the Sequential API ###
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

n_output_nodes = 3
model = Sequential()
dense_layer = tf.keras.layers.Dense(3, activation='sigmoid')
model.add(dense_layer)
x_input = tf.constant([[1, 2.]], shape=(1, 2))
model.build(tf.shape(x_input).numpy())
model.compile(optimizer = 'sgd', loss = 'mse')
model_output = model.predict(x_input)
print(model_output)

print('\n')

### Defining a model using subclassing ###
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SubclassModel(tf.keras.Model):

    def __init__(self, n_output_nodes):
        super(SubclassModel, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs):
        return self.dense_layer(inputs)

n_output_nodes = 3
model = SubclassModel(n_output_nodes)
x_input = tf.constant([[1, 2.]], shape = (1, 2))
print(model.call(x_input))

print('\n')

class IdentityModel(tf.keras.Model):

    def __init__(self, n_output_nodes):
        super(IdentityModel, self).__init__()
        self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')

    def call(self, inputs, isidentity=False):
        x = self.dense_layer(inputs)
        if isidentity == True:
            return inputs
        return x

n_output_nodes = 3
model = IdentityModel(n_output_nodes)
x_input = tf.constant([[1, 2.]], shape=(1, 2))
out_activate = model.call(x_input)
out_identity = model.call(x_input, True)
print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))

print('\n')


###
#### 1.4 Automatic differentiation in TensorFlow ####
###

### Gradient computation with GradientTape ###
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x * x
dy_dx = tape.gradient(y, x)
assert dy_dx.numpy() == 6


### Function minimization with automatic differentiation and SGD ###
x = tf.Variable([tf.random.normal([1])])    # Random value for x (normal distribution shape=1)
print("Initializing x = {}".format(x.numpy()))
learning_rate = 1e-2    # Learning rate for SGD
history = []
x_f = 4     # Target value

# Use of SGD for a number of iterations. Each, we compute loss, derivative of the loss with respect to x and SGD update
for i in range(500):
    with tf.GradientTape() as tape:
        loss = (x - x_f) ** 2
    grad = tape.gradient(loss, x)
    new_x = x - learning_rate * grad
    x.assign(new_x)
    history.append(x.numpy()[0])

plt.plot(history)
plt.plot([0, 500], [x_f, x_f])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')