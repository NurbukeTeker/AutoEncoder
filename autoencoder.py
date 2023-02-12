from keras.layers import Input, Dense
from keras.models import Model

# Define the dimensions of the input and encoding layers
encoding_dim = 32
input_dim = 784

# Define the input layer
input_layer = Input(shape=(input_dim,))

# Define the encoding layer
encoded = Dense(encoding_dim, activation='relu')(input_layer)

# Define the decoding layer
decoded = Dense(input_dim, activation='sigmoid')(encoded)

# Combine the input layer and the decoded layer to create the autoencoder
autoencoder = Model(input_layer, decoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
