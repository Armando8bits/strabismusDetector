import cv2
import mediapipe as mp
import time
import os

def detectar_desviacion(face_landmarks):
    altura_ojos = [face_landmarks.landmark[i].y for i in [159, 145, 386, 374]]
    ojo_izq_centro = (altura_ojos[0] + altura_ojos[1]) / 2
    ojo_der_centro = (altura_ojos[2] + altura_ojos[3]) / 2
    return abs(ojo_izq_centro - ojo_der_centro) > 0.02

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    cap = cv2.VideoCapture(0)
    
    fecha_actual = time.strftime("%Y%m%d")
    carpeta_destino = os.path.join(os.getcwd(), f"estrabismo_{fecha_actual}")
    os.makedirs(carpeta_destino, exist_ok=True)
    
    cv2.namedWindow("Detección de Estrabismo")  # Crear una ventana con nombre
    
    while True:  # Bucle infinito
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
        
        k = cv2.waitKey(1)
        if k == ord('q') or cv2.getWindowProperty("Detección de Estrabismo", cv2.WND_PROP_VISIBLE) < 1:
            break  # Salir si se presiona 'q' o se cierra la ventana
    
    cap.release()
    cv2.destroyAllWindows()

main()