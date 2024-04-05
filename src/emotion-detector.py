
import cv2 # pip install opencv-python
from keras.models import load_model
import numpy as np

def emotion_detector():
    # Chargement du modèle de détection des émotions
    model = load_model('../results/neural_network.keras')

    # Dictionnaire pour mapper les indices de classes aux émotions correspondantes
    # emotion_dict = {0: "Colère", 1: "Dégoût", 2: "Peur", 3: "Joyeux", 4: "Triste", 5: "Surpris", 6: "Neutre"}
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

    # Initialisation du détecteur de visages avec le modèle de Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Capture vidéo depuis la webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Conversion de l'image en niveaux de grisq
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages dans l'image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5, minSize=(48,48))
        
        for (x, y, w, h) in faces:
            # Extraction du visage de la région d'intérêt
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            
            # Normalisation des pixels
            roi = roi_gray / 255.0
            
            # Prédiction de l'émotion
            roi = np.expand_dims(roi, axis=0)
            roi = np.expand_dims(roi, axis=-1)
            prediction = model.predict(roi)
            
            # Affichage de l'émotion prédite
            maxindex = int(np.argmax(prediction))        
            # Taille de la zone de texte
        text_size = cv2.getTextSize(emotion_dict[maxindex], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

        # Dessiner un rectangle noir pour effacer l'ancien texte
        text_location = (50, 50)  # Coordonnées de l'emplacement où afficher le texte
        cv2.rectangle(frame, (text_location[0], text_location[1] - text_size[1]), (text_location[0] + text_size[0], text_location[1] + text_size[1]), (0, 0, 0), -1)

        # Affichage du texte de l'émotion prédite
        cv2.putText(frame, emotion_dict[maxindex], text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Affichage de la vidéo en direct avec les émotions détectées
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------- main ------------------------------------------
if __name__ == "__main__":
    emotion_detector()