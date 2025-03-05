import cv2
import mediapipe as mp
import time
import os

nombreAPP = "Detector de Estrabismo by Armando8bits"

def detectar_desviacion(face_landmarks):
    altura_ojos = [face_landmarks.landmark[i].y for i in [159, 145, 386, 374]]
    ojo_izq_centro = (altura_ojos[0] + altura_ojos[1]) / 2
    ojo_der_centro = (altura_ojos[2] + altura_ojos[3]) / 2
    return abs(ojo_izq_centro - ojo_der_centro) > 0.02 #el umbral de desviación debe ser ajusado según velocidad de cámara y luminosidad.

def detectar_parpadeo(face_landmarks):
    # Distancia vertical entre los párpados
    distancia_ojo_izq = abs(face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
    distancia_ojo_der = abs(face_landmarks.landmark[386].y - face_landmarks.landmark[374].y)
    # Umbral para detectar parpadeo (ajusta según sea necesario)
    umbral_parpadeo = 0.005
    return distancia_ojo_izq < umbral_parpadeo or distancia_ojo_der < umbral_parpadeo

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    
    fecha_actual = time.strftime("%Y%m%d")
    carpeta_destino = os.path.join(os.getcwd(), f"estrabismo_{fecha_actual}")
    os.makedirs(carpeta_destino, exist_ok=True)
    
    cv2.namedWindow(nombreAPP)  # Crear una ventana con nombre
    
    while True:  # Bucle infinito
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if not detectar_parpadeo(face_landmarks) and detectar_desviacion(face_landmarks):
                    timestamp = time.strftime("%H%M%S")
                    filename = os.path.join(carpeta_destino, f"estrabismo_{timestamp}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Desviación detectada, imagen guardada como {filename}")
                    
        cv2.imshow(nombreAPP, frame)
        
        k = cv2.waitKey(1)
        if k == ord('q') or cv2.getWindowProperty(nombreAPP, cv2.WND_PROP_VISIBLE) < 1:
            break  # Salir si se presiona 'q' o se cierra la ventana
    
    cap.release()
    cv2.destroyAllWindows()

main()