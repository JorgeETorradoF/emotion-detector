from deepface import DeepFace
import cv2
import numpy as np

def obtener_emocion_dominante(result):
    return result[0]['dominant_emotion']

# Iniciar cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo acceder a la cámara")
else:
    print("Presiona 'q' para salir")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        #Lo pasamos a escala de grises y luego a RGB
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        # Analizamos emoción en el frame actual con el Deepface
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='yolov8',align=True)

        # Enmarcamos el rostro y sacamos la emoción dominante (la de más puntaje)
        face_region = result[0]['region']
        x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
        emocion = obtener_emocion_dominante(result)

        # Dibujamos rectángulo (verde)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dibujar el texto con la emoción (ajustar posición si es necesario)
        cv2.putText(frame, emocion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    except Exception as e:
        print("Error:", e)

    # Mostrar frame
    cv2.imshow('Detección de Emociones', frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
