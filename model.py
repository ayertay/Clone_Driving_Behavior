import csv
import os
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn
import math

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)
    
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement*-1.0)
    
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
#train_samples, validation_samples = train_test_split(lines, test_size=0.2)

"""        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
 
            images = []
            measurements = []
            correction = 0.2
            for batch_sample in batch_samples:
                
                center_path = batch_sample[0]
                left_path = batch_sample[1]
                right_path = batch_sample[2]
                
                filename_c = center_path.split('\\')[-1]
                filename_l = left_path.split('\\')[-1]
                filename_r = right_path.split('\\')[-1]
                
                center_path = 'my_data/IMG/' + filename_c
                left_path = 'my_data/IMG/' + filename_l
                right_path = 'my_data/IMG/' + filename_r
                
                center_image = ndimage.imread(center_path)
                left_image = ndimage.imread(left_path)
                right_image = ndimage.imread(right_path)
                #images.extend([center_image, left_image, right_image])
                images.append(center_image)
                center_measurement = float(line[3])
                # create adjusted steering measurements for the side camera images
                steering_left = center_measurement + correction
                steering_right = center_measurement - correction
                #measurements.extend([center_measurement, steering_left, steering_right])
                measurements.append(center_measurement)
                
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(np.fliplr(image))
                augmented_measurements.append(measurement*-1.0)
            # trim image to only see section with road
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=128

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
"""
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
model.save('model.h5')
"""
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size), validation_data=validation_generator, nb_val_samples=math.ceil(len(validation_samples)/batch_size), nb_epoch=7, verbose=1)
"""
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

