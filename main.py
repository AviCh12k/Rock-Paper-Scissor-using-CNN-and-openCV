# import cv2
# import numpy as np
# from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
# from keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img

# Initializing CNN
classifier = Sequential()

# step-1 Convolution
classifier.add(Convolution2D(32, kernel_size=(3,3), input_shape=(64,64,3), activation= "relu"))

# step-2 pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step-3 Flattening
classifier.add(Flatten())

# Step-4 Full Connection
classifier.add(Dense(128, activation = "relu"))
classifier.add(Dense(1, activation = "softmax"))

#Compiling The CNN
classifier.compile(optimizer = "adam", loss= "categorical_crossentropy", metrics=['categorical_accuracy'])

# Part-2 Fitting thr=e CNN into 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Rock-Paper-Scissors/train',
        target_size=(32, 3),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        'Rock-Paper-Scissors/test',
        target_size=(128, 1),
        batch_size=32,
        class_mode='categorical')

from keras.callbacks import History
history = History()


classifier.fit(
        train_generator,
        steps_per_epoch=2520,
        epochs=25, 
        )






# =============================================================================
# cap = cv2.VideoCapture(0)
# while True:
#     _,frame = cap.read()
#     cv2.imshow("out",frame)
#     h,w,_ = frame.shape
#     # roi_mask = np.zeros([h,w])
#     # print(h,w)
#     roi = frame[:300,:320]
#     cv2.imshow("mask",roi)
#     
# 
# 
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break
# cap.release()
# cv2.destroyAllWindows()
# =============================================================================
