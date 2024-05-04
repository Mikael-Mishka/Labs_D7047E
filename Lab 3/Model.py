import keras
from keras.models import Model
from keras.layers import Embedding, Dense, Activation, MaxPool1D, Input, LSTM, Dropout, Input, Activation, add, MaxPooling2D, Conv2D, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2

def Convolution(input_tensor,filters):
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1),kernel_regularizer=l2(0.003))(input_tensor)
    x = Dropout(0.1)(x)
    x = Activation('relu')(x)
    return x

def model(input_shape):
  inputs_images=Input((input_shape))
  conv_1= Convolution(inputs_images,32)
  maxp_1 = MaxPooling2D(pool_size = (2,2)) (conv_1)
  conv_2 = Convolution(maxp_1,64)
  maxp_2 = MaxPooling2D(pool_size = (2, 2)) (conv_2)
  conv_3 = Convolution(maxp_2,128)
  maxp_3 = MaxPooling2D(pool_size = (2, 2)) (conv_3)
  flatten= Flatten() (maxp_3)
  dense_1= Dense(256,activation='relu')(flatten)

  # TODO: lstm layers
  
  # TODO: combination layers

  mdl= Model(inputs=[inputs_images],outputs=[dense_1]) # TODO: change dense with the output of the combination, add inputs from words
  mdl.summary()
  
  mdl.compile(loss="categorical_crossentropy", optimizer="Adam")

  return mdl

#base_model = VGG16(weights='imagenet', include_top=False)
#image_features = base_model.predict(img_array)
#image_features = image_features.reshape(image_features.shape[0], -1)

