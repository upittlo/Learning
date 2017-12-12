# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:29:31 2017

@author: Vincent.EC.Lo
"""

#%%
### Create 1d tensor

import numpy as np
tensor_1d = np.array([1.3,1,4.0,23.99])

print(tensor_1d)

print(tensor_1d[0])
print(tensor_1d[1])

### Check the rank 

print("rank = ",tensor_1d.ndim)
print("shape = ",tensor_1d.shape)
print("datatype = ",tensor_1d.dtype)
#%%
#### Convert to tensor
### use tf.convert_to_tensor(variable,dtype)
import tensorflow as tf


tf_tensor = tf.convert_to_tensor(tensor_1d,dtype = tf.float64)

with tf.Session() as sess:
    print(sess.run(tf_tensor))
    print(sess.run(tf_tensor[0]))
#%%

#### Create 2d array

#### tensor[row,column]
tensor_2d = np.array([(1,2,3,4),(4,5,6,7),(8,9,10,11),(12,13,14,15)])
print(tensor_2d)

tensor_2d[0:2,0:2]

#%%

matrix1 = np.array([(2,2,2),(2,2,2),(2,2,2)],dtype = 'int32')
matrix2 = np.array([(1,1,1),(1,1,1),(1,1,1)],dtype = 'int32')

print('matrix1 = ',matrix1)
print('matrix2 = ',matrix2)

matrix1 = tf.constant(matrix1)
matrix2 = tf.constant(matrix2)

### tensorflow operators
matrix_product = tf.matmul(matrix1,matrix2)
matrix_sum = tf.add(matrix1,matrix2)

matrix_3 = np.array([(2,7,2),(1,4,2),(9,0,2)],dtype = 'float32')

print("matrix3 = ")
print(matrix_3)

### 行列式

matrix_det = tf.matrix_determinant(matrix_3)

## execute

with tf.Session() as sess:
    result1 = sess.run(matrix_product)
    result2 = sess.run(matrix_sum)
    result3 = sess.run(matrix_det)
    
print(result1)
print(result2)
print(result3)


#%%

### 3d tensor

tensor_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(tensor_3d)

print("shape = ",tensor_3d.shape)

### structure  tensor_3d[plane,row,column]

print("shape = ",tensor_3d[0,1,0])
#%%
#### import a picture

import matplotlib.image as mp_image

filename = "packt.jpeg"

input_image = mp_image.imread(filename)

print("input dim = {}".format(input_image.ndim))

print("input shape = {}".format(input_image.shape))

#%%

### use matplot to show image

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

#%%

my_image = tf.placeholder("uint8",[None,None,3])


### slice the image

slice = tf.slice(my_image,[10,0,0],[16,-1,-1])

with tf.Session() as session:
    result = session.run(slice,feed_dict = {my_image : input_image})
    print(result.shape)
plt.imshow(result)
plt.show()

#%%

## image Transpose

x = tf.Variable(input_image,name ='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x, perm = [1,0,2])
    session.run(model)
    result  =session.run(x)
    
plt.imshow(result)
plt.show()


#%%

#### Calculate gradient

import tensorflow as tf

x = tf.placeholder(tf.float32)

y =2*x*x

var_grad = tf.gradients(y,x)

with tf.Session() as sess:
    var_grad_val = sess.run(var_grad,feed_dict = {x:1})
    
print(var_grad_val)

#%%

#### Generate Random Variable

## 1. uniform distribution

# random_uniform(shape, minval, maxval, dtype, seed,name)

uniform = tf.random_uniform([100],minval = 0,maxval  =1,dtype = tf.float32)

with tf.Session() as sess:
    print(uniform.eval())
    plt.hist(uniform.eval(),normed = True)
    plt.show()



#%%
#### Normal distribution

norm = tf.random_normal([100],mean = 0,stddev = 2)
with tf.Session() as sess:
    plt.hist(norm.eval(),normed = True)
    plt.show()


#%%
### Pseudo random and random 

uniform_with_seed = tf.random_uniform([1],seed =1)
uniform_without_seed = tf.random_uniform([1])

print("First run")
with tf.Session() as first_session:
    print("uniform with (seed = 1) = {}".format(first_session.run(uniform_with_seed)))
    
    print("uniform with (seed = 1) = {}".format(first_session.run(uniform_with_seed)))
    print("uniform with (seed = 1) = {}".format(first_session.run(uniform_with_seed)))
    print("uniform with (seed = 1) = {}".format(first_session.run(uniform_with_seed)))
    
print("Second run")

with tf.Session() as second_session:
    print("uniform with (seed = 1) = {}".format(second_session.run(uniform_with_seed)))
    print("uniform with (seed = 1) = {}".format(second_session.run(uniform_with_seed)))
    print("uniform with (seed = 1) = {}".format(second_session.run(uniform_without_seed)))
    print("uniform with (seed = 1) = {}".format(second_session.run(uniform_without_seed)))
    
    
    



#%%

#### Monte Carlo

trials = 100
hits = 0

### Generate random number between -1 to 1

x=  tf.random_uniform([1],minval = -1,maxval = 1,dtype = tf.float32)
y = tf.random_uniform([1],minval = -1,maxval = 1,dtype = tf.float32)

pi = []

sess = tf.Session()

with sess.as_default():
    for i in range(1,trials):
        for j in range(1,trials):
            if x.eval()**2 + y.eval()**2<1:
                hits = hits+1
                pi.append((4*float(hits)/i)/trials)
                
plt.plot(pi)
plt.show()


#%%

##### 雨滴落在池子中
# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
sess = tf.InteractiveSession()

N = 500

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(100):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

plt.imshow(u_init)
plt.show()
#%%

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

# Discretized PDE update rules
U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

# Operation to update the state
step = tf.group(
  U.assign(U_),
  Ut.assign(Ut_))

# Initialize state to initial conditions
tf.initialize_all_variables().run()

# Run 1000 steps of PDE
for i in range(1000):
  # Step simulation
  step.run({eps: 0.03, damping: 0.04})
  # Visualize every 50 steps
  if i % 500 == 0:
      clear_output()
      plt.imshow(U.eval())
      plt.show()





#%%


