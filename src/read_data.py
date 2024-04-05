import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical # one-hot-encoding

#  emotion statistique
def count_emotions(y):
    """
    affiche le nombre de chaque émotion
    Args:
        y (numpy.ndarray): émotions
    """
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    
    # Convertir les catégories encodées en indices d'émotion
    y_emotions = np.argmax(y, axis=1)
    
    # Compter les émotions
    counts = np.bincount(y_emotions)
    
    # Afficher le nombre d'occurrences de chaque émotion
    for i, emotion in enumerate(emotion_dict):
        print(f"{emotion_dict[emotion]}: {counts[i]}")

def emotion_graph(y_train, y_test):
    """
    Créé un graphique pour les émotions extraite des données.
    Args:
        y_train (numpy.ndarray): emotions Train
        y_test (numpy.ndarray): emotions Test
    """
    # Émotions disponibles
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)

    # Créer l'histogramme Train
    axs[0].hist(y_train.argmax(axis=1), bins=np.arange(len(emotions) + 1) - 0.5, color='skyblue', label=emotions, alpha=0.7, rwidth=0.8)
    axs[0].set_xticks(np.arange(len(emotions)))
    axs[0].set_xticklabels(emotions, rotation=45)
    axs[0].set_xlabel('Émotions')
    axs[0].set_ylabel('Fréquence')
    axs[0].set_title('Répartition des émotions pour les données Train')

    # Créer l'histogramme Test
    axs[1].hist(y_test.argmax(axis=1), bins=np.arange(len(emotions) + 1) - 0.5, color='lightgreen', label=emotions, alpha=0.7, rwidth=0.8)
    axs[1].set_xticks(np.arange(len(emotions)))
    axs[1].set_xticklabels(emotions, rotation=45)
    axs[1].set_xlabel('Émotions')
    axs[1].set_ylabel('Fréquence')
    axs[1].set_title('Répartition des émotions pour les données Test')

    plt.show()

# Préparation data
def reconstitute_images(X):
    """
    Divise chaque chaîne de pixels en une liste de valeurs individuelles, convertit ces valeurs en flottants et les normalise en 
    les divisant par 255.
    Puis regroupe ces listes de pixels dans un tableau, et reconstitue en matrices 48x48x1.

    Args:
        X (str): valeurs des pixels des images sous forme de chaîne de caractères

    Returns:
        tableau numpy: image
    """
    print("--------- reconstitute_images ! -----------------")
    
    # Reconstitution et normalisation des images à partir des pixels.
    pixel_lists = X.str.split().apply(lambda x: [float(i)/255 for i in x])

    # Matrice 48x48
    pixel_arrays = np.array(pixel_lists.tolist())
    image = pixel_arrays.reshape(-1, 48, 48, 1) 
    
    return image

def read_data():
    """
    Récupération des images:
    Lecture des données par lot de 1000 lignes pour éviter de saturer la mémoire au moment de la reconstitution d'image.
    
    """
    print("---------------- read_data ! -------------------")
    
    # Listes pour stocker les données
    X_train_list = []
    X_test_list = []
    y_train_list = []
    y_test_list = []
    
    # Variables pour stocker les pixels de la dernière image de chaque lot
    last_train_image_pixels = None
    last_test_image_pixels = None

    batch_size = 1000  # Nombre de lignes à lire à la fois
    
    # itération sur chaque lot de 1000 lignes
    for train_chunk, test_chunk in zip(pd.read_csv('../data/train.csv', chunksize=batch_size), 
                                        pd.read_csv('../data/test_with_emotions.csv', chunksize=batch_size)):

        # Vérification des valeurs manquantes dans les données. (pour debug)
        # print("train_chunk :")
        # print(train_chunk.isnull().sum())
        # print("test_chunk :")
        # print(test_chunk.isnull().sum())

        # Reconstitution des images
        X_train = reconstitute_images(train_chunk['pixels'])
        X_test = reconstitute_images(test_chunk['pixels'])
        
        # Stocker les pixels de la dernière image de chaque lot. (debug pour afficher la dernière aprés la boucle)
        last_train_image_pixels = X_train[-1]  # Dernière image du lot de train
        last_test_image_pixels = X_test[-1]
        
        # Convertion des étiquettes (émotions) en un format approprié pour l'entraînement.
        y_train = to_categorical(train_chunk['emotion'], num_classes=7)
        y_test = to_categorical(test_chunk['emotion'], num_classes=7)
        
        # Ajouter les étiquettes actuelles à la liste
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)

    # Afficher la dernière image Train après la boucle (pour debug)
    plt.imshow(last_train_image_pixels[:,:,0], cmap='gray')
    plt.show()

    # Afficher la dernière image Test après la boucle (pour debug)
    plt.imshow(last_test_image_pixels[:,:,0], cmap='gray')
    plt.show()

    # # Libérer l'espace mémoire
    del last_train_image_pixels
    del last_test_image_pixels
    
    # Concaténer les listes pour obtenir des tableaux complets
    X_train = np.concatenate(X_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    #  Afficher les nombres d'émotion Train et Test
    print("--- Train emotions: ---")
    count_emotions(y_train)
    print("--- Test emotions: ---")
    count_emotions(y_test)
    
    emotion_graph(y_train, y_test)

    return X_train, X_test, y_train, y_test


# ---------------------------------------- main ------------------------------------------
if __name__ == "__main__":
    read_data()