from uagents import Agent, Context
import cv2
import time
import mediapipe as mp
import numpy as np
import reconocimiento
import ejecucion

agent = Agent(name="agent", seed="lsch")

# Inicializar MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

@agent.on_event("startup")
async def say_hello(ctx: Context):
    # Capturar el video desde la cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return
    
    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        signs = []
        while cap.isOpened():
            # Leer un frame
            ret, frame = cap.read()
            
            # Convertir el frame a RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Hacer la detección
            results = holistic.process(image)
            if results.right_hand_landmarks:
                kp = extract_keypoints(results)
                signs.append(kp)
                sign = reconocimiento.recon(kp)
                print(sign)
                ejecucion.controlador_accion(sign)
                
            
            # Convertir la imagen de nuevo a BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Dibujar las anotaciones de pose de MediaPipe
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            
            # Mostrar el frame con las anotaciones
            cv2.imshow('MediaPipe Holistic', image)
            
            # keypoints = extract_keypoints(results)
            # print(keypoints)
            
            
            # Salir con la tecla 'q'
            if cv2.waitKey(1) == ord('q'):
                np.save('./right.npy', signs)
                break
    
    # Liberar la captura de video
    cap.release()
    cv2.destroyAllWindows()


def extract_keypoints(result):
    RH_X = np.array([s.x for s in result.right_hand_landmarks.landmark]) if result.right_hand_landmarks else np.zeros(21)
    RH_Y = np.array([s.y for s in result.right_hand_landmarks.landmark]) if result.right_hand_landmarks else np.zeros(21)
    temp = RH_X, RH_Y
    keypoints = np.concatenate(temp)
    np_keypoints = np.array(keypoints)
    return np_keypoints



if __name__ == "__main__":
    agent.run()








