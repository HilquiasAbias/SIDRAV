# import cv2

# # Inicialize a captura de vídeo da webcam
# cap = cv2.VideoCapture(0)

# # Verifique se a captura foi iniciada com sucesso
# if not cap.isOpened():
#   print("Erro ao abrir a câmera")
#   exit()

# # Defina o codec e crie o objeto VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# while True:
#   ret, frame = cap.read()
  
#   if not ret:
#     print("Não foi possível ler o frame")
#     break

#   # Escreva o frame no arquivo de saída
#   out.write(frame)

#   # Mostre o frame na tela
#   cv2.imshow('frame', frame)

#   # Pressione 'q' para sair
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# # Libere o objeto de captura e feche todas as janelas
# cap.release()
# out.release()
# cv2.destroyAllWindows()


import cv2
import threading
import time
import numpy as np

# Variável global para indicar quando a captura de vídeo deve parar
stop_capture = False

def capture_video():
    global stop_capture
    
    # Inicialize a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)

    # Verifique se a captura foi iniciada com sucesso
    if not cap.isOpened():
        print("Erro ao abrir a câmera")
        return

    # Defina o codec e crie o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while not stop_capture:
        ret, frame = cap.read()

        if not ret:
            print("Não foi possível ler o frame")
            break

        # Escreva o frame no arquivo de saída
        out.write(frame)

        # Mostre o frame na tela
        cv2.imshow('frame', frame)

        # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_capture = True
            break

    # Libere o objeto de captura e feche todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_video():
    global stop_capture

    while not stop_capture:
        # Aguardar X segundos antes de processar a cópia do vídeo
        time.sleep(20)  # Ajuste este tempo conforme necessário

        # Reabra o arquivo de vídeo para processar
        cap = cv2.VideoCapture('output.avi')

        # Verifique se o arquivo de vídeo foi aberto com sucesso
        if not cap.isOpened():
            print("Erro ao abrir o arquivo de vídeo")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Realize a transformação no frame (por exemplo, converter para escala de cinza)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            sobel_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente na direção x
            sobel_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente na direção y

            # Magnitude do gradiente
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            magnitude = np.uint8(magnitude)  # Converter para uint8

            # Aplicar o detector de Canny
            bordas = cv2.Canny(gray_frame, threshold1=100, threshold2=200)  # Limiares inferior e superior

            # Mostre o frame transformado (opcional)
            cv2.imshow('Processed frame', bordas)

            # Pressione 'q' para sair da visualização do frame processado
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_capture = True
                break

        cap.release()

# Crie e inicie as threads
capture_thread = threading.Thread(target=capture_video)
process_thread = threading.Thread(target=process_video)

capture_thread.start()
process_thread.start()

# Espere as threads terminarem
capture_thread.join()
process_thread.join()

# Feche todas as janelas ao final
cv2.destroyAllWindows()
