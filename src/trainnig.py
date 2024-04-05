
import numpy as np
import matplotlib.pyplot as plt

# build CNN
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from read_data import read_data

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# """
#     La ligne de code os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' est une instruction Python qui définit une variable d'environnement pour 
#     TensorFlow. Cette ligne de code configure le niveau de journalisation (logging) de TensorFlow.

#     En particulier, cette instruction définit TF_CPP_MIN_LOG_LEVEL à '3', ce qui signifie que seules les erreurs critiques seront 
#     affichées dans la sortie de journalisation (logs) de TensorFlow. Cela signifie que les messages d'information, de débogage et de 
#     mise en garde ne seront pas affichés.

#     Cela est souvent utilisé pour supprimer les messages de journalisation inutiles ou indésirables lors de l'exécution de scripts 
#     TensorFlow, ce qui peut rendre la sortie plus propre et plus facile à analyser. Cependant, cela peut également masquer des 
#     informations importantes de débogage, il est donc important de l'utiliser avec discernement, en fonction des besoins spécifiques 
#     de votre application.
# """

# -------------------------------------------------------------------------------------------------------------

# Convolutional Neural Network
def Conv3x3(filter):
    # Construction du modèle CNN :
    return Conv2D(filters=filter, kernel_size=(3,3), padding="Same", activation="relu")

# -------------------------------------------------------------------------------------------------------------

# ----------------- VGG Neural Network ------------------------
def VGG_Neural_Network(network):
    PoolLayer = MaxPool2D(pool_size=(2,2), strides=(2,2))
    
    input_conv = Conv2D(
        filters=64, 
        kernel_size=(3,3), 
        padding="Same", 
        activation="relu", 
        input_shape=(48,48,1)
        )

    network.add(input_conv)
    network.add(Conv3x3(64))
    network.add(PoolLayer)
    network.add(Dropout(0.25))

    network.add(Conv3x3(128))
    network.add(Conv3x3(128))
    network.add(PoolLayer)
    network.add(Dropout(0.25))

    network.add(Conv3x3(256))
    network.add(Conv3x3(256))
    network.add(Conv3x3(256))
    network.add(PoolLayer)
    network.add(Dropout(0.25))


    network.add(Conv3x3(512))
    network.add(Conv3x3(512))
    network.add(Conv3x3(512))
    network.add(PoolLayer)
    network.add(Dropout(0.25))

    # Fully Connected Layers
    network.add(Flatten())
    network.add(Dense(4096, activation="relu"))
    network.add(Dense(4096, activation="relu"))
    network.add(Dropout(0.25))
    network.add(Dense(7, activation="softmax"))
    
    return network

# ------------------------- CNN Architecture (Digit Recognizer Modified) -------------------------------

def CNN_Architecture(network):
    ConvLayer32 = Conv2D(
        filters=32,
        kernel_size=(5,5),
        padding="Same",
        activation='relu'
    )

    ConvLayer64 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding="Same",
        activation='relu',
    )

    PoolLayer1 = MaxPool2D(pool_size=(2,2))
    PoolLayer2 = MaxPool2D(pool_size=(2,2), strides=(2,2))

    # Convolution/Pooling
    network.add(ConvLayer32)
    network.add(ConvLayer32)
    network.add(PoolLayer1)
    network.add(Dropout(rate=0.25))

    network.add(ConvLayer64)
    network.add(ConvLayer64)
    network.add(PoolLayer2)
    network.add(Dropout(rate=0.25))

    # Fully Connected
    network.add(Flatten()) # convert the 2D matrix into a 1D vector
    network.add(Dense(4096, activation="relu"))
    network.add(Dense(4096, activation="relu"))
    #network.add(Dropout(0.5))
    network.add(Dense(7, activation="softmax"))
    
    return network

# ---------------------------- The medium model -------------------------------------

def medium_model(network):
    input_conv = Conv2D(filters=32, kernel_size=(3,3), padding="Same", activation="relu", input_shape=(48,48,1))

    network.add(input_conv)
    network.add(MaxPool2D(pool_size=(2,2)))
    network.add(Dropout(0.1))

    network.add(Conv3x3(64))
    network.add(MaxPool2D(pool_size=(2,2)))
    network.add(Dropout(0.1))

    network.add(Conv3x3(128))
    network.add(MaxPool2D(pool_size=(2,2)))
    network.add(Dropout(0.1))

    network.add(Conv3x3(256))
    network.add(MaxPool2D(pool_size=(2,2)))
    network.add(Dropout(0.1))

    network.add(Flatten())
    network.add(Dense(units=128, activation='relu'))
    network.add(Dropout(0.2))
    network.add(Dense(7, activation='softmax'))
    
    return network


# ----------------------------- The FC-less model ----------------------------------

def FC_less_model(network):
    input_layer = Conv2D(32, (3,3), padding="same", activation='relu', input_shape=(48,48,1))
    network.add(input_layer)
    network.add(Dropout(0.3))

    network.add(Conv2D(64, (3,3), padding="same", activation='relu'))
    network.add(MaxPool2D(pool_size=(2,2)))
    network.add(Dropout(0.3))

    network.add(Flatten())
    network.add(Dense(7, activation='softmax'))
    
    return network

# -------------------------------------------------------------------------------------------------------------

def training():
    X_train, X_test, y_train, y_test = read_data()

    network = Sequential()
    # network = VGG_Neural_Network(network)
    # network = CNN_Architecture(network)
    network = medium_model(network)
    # network = FC_less_model(network)
    print(network.summary())

    optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08) 

    # network.compile(
    #     optimizer = 'adam' ,
    #     loss = "categorical_crossentropy",
    #     metrics=["accuracy"]
    # )
    network.compile(
        optimizer = optimizer ,
        loss = "categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ---
    # history_obj = network.fit(
    #     X_train, y_train,
    #     validation_data=(X_test, y_test),
    #     epochs=40,
    #     batch_size=64
    # )
    history_obj = network.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=25,
        batch_size=86
    )

    # ---
    # Evaluation
    history = history_obj.history

    loss_values = history['loss']
    val_loss_values = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    epochs = range(1, len(loss_values) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    #
    # Plot the model accuracy vs Epochs
    #
    ax[0].plot(epochs, accuracy,marker='o', color='royalblue', label='Training accuracy')
    ax[0].plot(epochs, val_accuracy, marker='o', color='orangered', label='Validation accuracy')
    ax[0].set_title('Training & Validation Accuracy', fontsize=16)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Accuracy', fontsize=16)
    ax[0].legend()
    #
    # Plot the loss vs Epochs
    #
    ax[1].plot(epochs, loss_values, marker='o', color='royalblue', label='Training loss')
    ax[1].plot(epochs, val_loss_values, marker='o', color='orangered', label='Validation loss')
    ax[1].set_title('Training & Validation Loss', fontsize=16)
    ax[1].set_xlabel('Epochs', fontsize=16)
    ax[1].set_ylabel('Loss', fontsize=16)
    ax[1].legend()

    best_epoch_acc = accuracy.index(np.max(accuracy))
    best_epoch_loss = loss_values.index(np.min(loss_values))

    print('accuracy inflection point : ', best_epoch_acc)
    print('loss inflection point :', best_epoch_loss)

    # Sauvegarde du modèle :
    network.save('../results/neural_network.keras')

    # Prédiction :
    proba_lists = network.predict(X_train)
    print("--- proba_lists: ---")
    print(proba_lists)

    # Evaluation
    loss_and_acc = network.evaluate(X_test, y_test)
    print("--- loss_and_acc: ---")
    print(loss_and_acc)


# --------------------------------- main ---------------------------------------------

if __name__ == "__main__":
    training()