#!/usr/bin/env python
# coding: utf-8

# # Modified LeNet Network 
# ## Classification Of Cheetah, Jaguar and Leopard on the basis of LeNet-5 Network Architecture
# 
# #### **INPUT Layer**
# The first is the data INPUT layer. The size of the input image is uniformly normalized to 64 * 64.
# 
# #### **C1 layer-convolutional layer**
# >**Input picture**: 64 * 64
# 
# >**Convolution kernel size**: 5 * 5
# 
# >**Convolution kernel types**: 6
# 
# >**Output featuremap size**: 60 * 60 (64-5 + 1) = 60
# 
# >**Number of neurons**: 60 * 60 * 6
# 
# >**Trainable parameters**: (5 * 5 + 1) * 6 (5 * 5 = 25 unit parameters and one bias parameter per filter, a total of 6 filters)
# 
# >**Number of connections**: (5 * 5 + 1) * 6 * 60 * 60 = 5,61,600
# 
# #### **S2 layer-pooling layer (downsampling layer)**
# 
# >**Input**: 60 * 60
# 
# >**Sampling area**: 3 * 3
# 
# >**Sampling method**: 4 inputs are added, multiplied by a trainable parameter, plus a trainable offset. Results via relu
# 
# >**Sampling type**: 6
# 
# >**Output featureMap size**: 30 * 30 (60/2)
# 
# >**Number of neurons**: 30 * 30 * 6
# 
# >**Trainable parameters**: 3 * 6 (the weight of the sum + the offset)
# 
# >**Number of connections**: (3 * 3 + 1) * 6 * 30 * 30
# 
# 
# #### **C3 layer-convolutional layer**
# 
# >**Input**: all 6 or several feature map combinations in S2
# 
# >**Convolution kernel size**: 3 * 3
# 
# >**Convolution kernel type**: 16
# 
# >**Output featureMap size**: 28 * 28 (30-3 + 1) = 28
# 
# 
# #### **S4 layer-pooling layer (downsampling layer)**
# 
# >**Input**: 28 * 28
# 
# >**Sampling area**: 2 * 2
# 
# >**Sampling type**: 16
# 
# >**Output featureMap size**: 14 * 15 (28/2)
# 
# #### **F6 layer-fully connected layer**
# 
# >**Input**: c3 120-dimensional vector
# 
# >**Calculation method**: calculate the dot product between the input vector and the weight vector, plus an offset, and the result is output through the sigmoid function.
# 
# >**Trainable parameters**: 84 * (120 + 1) = 10164
# 
# 
# 

# # Code Implementation

# In[1]:


import keras


# In[2]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[18]:


classifier = Sequential()
#Select 6 Convolution of size 3*3 , Input size of image is 32*32*3, it is a RGB image
classifier.add(Conv2D(6, kernel_size=(5,5), activation='relu', input_shape=(64,64, 3)))
#The output of the Convolution layer is 60*60*6 
#Trainable parameters is (5 * 5 + 1) * 6= 156; 
#(5 * 5 = 25 unit parameters and one bias parameter per filter, a total of 6 filters)

classifier.add( MaxPooling2D( pool_size=(3,3)))
#The output of the Maximum Pooling layer is 30*30*6

#The input matrix size of this layer is 30 * 30 * 6, the filter size used is 3 * 3, and the depth is 16. This layer does not use all 0 padding, and the step size is 1.
# The output matrix size of this layer is 28 * 28 * 16.
classifier.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
#The output of the Second Convolution layer is (30-3+1)=28
classifier.add( MaxPooling2D( pool_size=(2,2)))
#The output of the Maximum Pooling layer is 14*14*16
classifier.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
#The output of the Second Convolution layer is (14-5+1)=10; 10*10*16
classifier.add( MaxPooling2D( pool_size=(2,2)))
#The output of the Maximum Pooling layer is 5*5*16
# The input matrix size of this layer is 5 * 5 * 16. This layer is called a convolution layer in the LeNet-5 paper, but because the size of the filter is 5 * 5, #
# So it is not different from the fully connected layer. If the nodes in the 5 * 5 * 16 matrix are pulled into a vector, then this layer is the same as the fully connected layer.
# The number of output nodes in this layer is 120, with a total of 5 * 5 * 16 * 120 + 120 = 48120 parameters.
classifier.add(Flatten())
classifier.add(Dense(120, activation='relu'))

# The number of input nodes in this layer is 120 and the number of output nodes is 84. The total parameter is 120 * 84 + 84 = 10164 (w + b)
classifier.add(Dense(84, activation='relu'))

classifier.add(Dense(3, activation='softmax'))
classifier.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# In[19]:


# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator


train_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


test_data = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


training_set = train_data.flow_from_directory("C:/Users/Prateek's PC/Desktop/dataset/train",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_data.flow_from_directory("C:/Users/Prateek's PC/Desktop/dataset/test",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[20]:


classifier.fit_generator(training_set,
                         steps_per_epoch = 10,
                         epochs = 10,
                         validation_data = test_set,    
                         validation_steps = 10)


# In[27]:


# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img("C:/Users/Prateek's PC/Desktop/dataset/t1.jfif", target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)
if result[0][0] == 1:
    prediction = 'cheetah'
    print(prediction)
elif result[0][1] == 1:
    prediction = 'jaguar'
    print(prediction)
    
else:
    prediction = 'leopard'
    print(prediction)


# In[ ]:





# In[ ]:




