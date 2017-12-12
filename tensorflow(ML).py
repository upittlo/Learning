# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 13:58:05 2017

@author: Vincent.EC.Lo
"""

#%%

##### Linear Regression

import numpy as np

number_of_points = 500

x_point = []
y_point = []

a = 0.22
b = 0.78

### Generate 300 random points

for i in range(number_of_points):
    x = np.random.normal(0.0,0.5)
    y = a*x+b+np.random.normal(0.0,0.1)
    
    x_point.append([x])
    y_point.append([y])
#%%

import matplotlib.pyplot as plt
plt.plot(x_point,y_point,'o',label = 'Input data')
plt.legend()
plt.show()


#%%

### Use tensorflow to do linear regression

import tensorflow as tf

### Define variable
### 變數用 tf.Variable()
A = tf.Variable(tf.random_uniform([1],-1.0,1.0))
B=  tf.Variable(tf.zeros([1]))

y = A*x_point+B
#### reduce_mean(tensor,axis) >> get average

cost_function = tf.reduce_mean(tf.square(y-y_point)) 

## tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(cost_function)

model = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(model)
    for step in range(0,21):
        sess.run(train)
        
        if (step % 5) ==0:
            plt.plot(x_point,y_point,'o',label = 'step = {}'.format(step))
            plt.plot(x_point,sess.run(A)*x_point+sess.run(B)) 
            
            plt.legend()
            plt.show()


#%%

import tensorflow.examples.tutorials.mnist.input_data as input_data

#%%
import numpy as np
import matplotlib.pyplot as plt

mnist_images = input_data.read_data_sets("MNIST_data/",one_hot = False)
#%%


#train.next_batch(10)   
pixels,real_values = mnist_images.train.next_batch(10)  ### return first 10 image

### pixels (10,784) numpy array  >> x
### real_values (10,) numpy array >> y

print("list of values loaded",real_values)  ### This is an numpy array
example_to_visualize = 5
print("element N "+ str(example_to_visualize+1) + "of the list plotted")



#%%
#### plot the image

#### Data Structure: (number of examples,array)

image = pixels[example_to_visualize,:]

image = np.reshape(image,[28,28])
plt.imshow(image)
plt.plot()
#%%

## Classifier 
### Use 100 training data to train KNN
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

train_pixels,train_list_values = mnist.train.next_batch(300) 
test_pixels,test_list_of_values  = mnist.test.next_batch(10) 


train_pixel_tensor = tf.placeholder\
                     ("float", [None, 784])
test_pixel_tensor = tf.placeholder\
                     ("float", [784])

#Cost Function and distance optimization

distance = tf.reduce_sum\
           (tf.abs\
            (tf.add(train_pixel_tensor, \
                    tf.negative(test_pixel_tensor))), \
            reduction_indices=1)

pred = tf.arg_min(distance, 0)


#%%
###cost function

distance = tf.reduce_sum(tf.abs(tf.add(train_pixel_tensor,tf.negative(test_pixel_tensor))),
                         reduction_indices = 1)
### reduction_indices=1 >> axis = 1
### 沿著pixels 計算
### tf.reduce
### for example

'''
x = np.array([[1,1,1],[1,1,1]])
tf.reduce_sum(x) >> 6
tf.reduce_sum(x,0) >> [2,2,2]
tf.reduce_sum(x,1) >> [3,3]
tf.reduce_sum(x,1,keep_dims = True) >> [[3],[3]]
tf.reduce_sum(x,[0,1]) >> 6
'''

pred = tf.arg_min(distance,0)

accuracy = 0

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(test_list_of_values)):
        nn_index = sess.run(pred,\
		  feed_dict={train_pixel_tensor:train_pixels,\
		  test_pixel_tensor:test_pixels[i,:]})
        
        print("Test N° ", i,"Predicted Class: ", \
		  np.argmax(train_list_values[nn_index]),\
		"True Class: ", np.argmax(test_list_of_values[i]))
        if np.argmax(train_list_values[nn_index])\
		      == np.argmax(test_list_of_values[i]):
            accuracy += 1./len(test_pixels)
    print ("Result = ", accuracy)
            
        
    



#%%



