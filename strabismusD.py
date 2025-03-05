import cv2
import mediapipe as mp
import time
import os

def detectar_desviacion(face_landmarks):
    """
    Detecta la desviación de los ojos a partir de los puntos de referencia faciales.
    
    Args:
        face_landmarks: Los puntos de referencia faciales detectados por MediaPipe.
        
    Returns:
        True si se detecta desviación, False en caso contrario.
    """
    altura_ojos = [face_landmarks.landmark[i].y for i in [159, 145, 386, 374]]
    ojo_izq_centro = (altura_ojos[0] + altura_ojos[1]) / 2
    ojo_der_centro = (altura_ojos[2] + altura_ojos[3]) / 2
    return abs(ojo_izq_centro - ojo_der_centro) > 0.02  # Umbral para detectar desviación

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    
    # Crear carpeta con la fecha actual
    fecha_actual = time.strftime("%Y%m%d")
    carpeta_destino = os.path.join(os.getcwd(), f"estrabismo_{fecha_actual}")
    os.makedirs(carpeta_destino, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if detectar_desviacion(face_landmarks):
                    timestamp = time.strftime("%H%M%S")
                    filename = os.path.join(carpeta_destino, f"estrabismo_{timestamp}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Desviación detectada, imagen guardada como {filename}")
                    
        cv2.imshow("Detección de Estrabismo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main()