import cv2
import mediapipe as mp
#import numpy as np
import time

def detectar_estrabismo():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                altura_ojos = [face_landmarks.landmark[i].y for i in [159, 145, 386, 374]]
                ojo_izq_centro = (altura_ojos[0] + altura_ojos[1]) / 2
                ojo_der_centro = (altura_ojos[2] + altura_ojos[3]) / 2
                
                if abs(ojo_izq_centro - ojo_der_centro) > 0.02:  # Umbral para detectar desviación
                    filename = f"estrabismo_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, frame)
                    print(f"Desviación detectada, imagen guardada como {filename}")
                    
        cv2.imshow("Detección de Estrabismo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

detectar_estrabismo()
