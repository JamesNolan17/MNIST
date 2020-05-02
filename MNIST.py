# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution       特征探测器数目，大小（3*3矩阵）（64*64，3位色深）                  添加卷🐔层 
classifier.add(Convolution2D(32, 3, 3, input_shape = (28,28,3), activation = 'relu'))

# Step 2 - Pooling                       窗口大小2*2                                    添加最大池化层（降维）
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening                                                                 扁平化
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Compiling the CNN                    两个才能用binary 多个用categorical
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images  防止过度拟合---> imagedatagenerator


#                                   0-255-------> 0-1
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2, #错切
                                   zoom_range = 0.2,  #放大缩小
                                   horizontal_flip = True) #翻转

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('mnist_train',
                                                 target_size = (28,28),
                                                 batch_size = 32,  #每批生成几个
                                                 class_mode = 'categorical') #两个才能用binary 多个用categorical

test_set = test_datagen.flow_from_directory('mnist_train',
                                            target_size = (28, 28),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                          samples_per_epoch = 60000,
                          nb_epoch = 15,
                          validation_data = test_set,
                          nb_val_samples = 10000)

