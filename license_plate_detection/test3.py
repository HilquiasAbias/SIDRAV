# import cv2
# import threading
# import time
# import queue
# import numpy as np

# from main import LicensePlateDetector

# # Inicializa a captura da webcam
# cap = cv2.VideoCapture(0)

# # Define o codec e cria o objeto VideoWriter
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# # Lista para armazenar frames capturados
# frames = []
# processing_queue = queue.Queue()
# processing_started = False

# def process_frames():
#   global frames, processing_started
#   detector = LicensePlateDetector(video_path=None, cascade_path='UKChars33_16x25_11W.xml', debug=True)
#   while True:
#     if processing_started and frames:
#       # # Realiza alguma transformação nos frames
#       # frame = frames.pop(0)  # Pega o primeiro frame da lista
#       # # Exemplo de transformação: converte para escala de cinza
#       # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#       # sobel_x = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente na direção x
#       # sobel_y = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente na direção y

#       # # Magnitude do gradiente
#       # magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
#       # magnitude = np.uint8(magnitude)  # Converter para uint8

#       # # Aplicar o detector de Canny
#       # processed_frame = cv2.Canny(processed_frame, threshold1=100, threshold2=200)
#       # # Coloca o frame processado na fila
#       # processing_queue.put(processed_frame)
#       frame = frames.pop(0)
#       detector.process_frame(frame)

# # Inicia a thread de processamento
# thread = threading.Thread(target=process_frames)
# thread.start()

# # Captura e salva frames
# start_time = time.time()
# while True:
#   ret, frame = cap.read()
  
#   if not ret:
#     break

#   # Salva o frame no arquivo de vídeo
#   out.write(frame)
#   frames.append(frame)  # Adiciona o frame à lista

#   # Inicia o processamento após 10 segundos
#   if time.time() - start_time > 15:
#     processing_started = True

#   # Exibe frames processados na thread principal
#   while not processing_queue.empty():
#     processed_frame = processing_queue.get()
#     cv2.imshow('Processed Frame', processed_frame)

#   # Remoção de frames processados da lista
#   if processing_started and len(frames) > 0:
#     frames = frames[1:]  # Remove o primeiro frame processado

#   # Condição de parada: pressione 'q' para sair
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# # Libera a captura e fecha as janelas
# cap.release()
# out.release()
# cv2.destroyAllWindows()

import cv2
import threading
import time
import queue
import os
import shutil

from main import LicensePlateDetector

# Inicializa a captura da webcam
cap = cv2.VideoCapture(0)

# Define o codec e cria o objeto VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Lista para armazenar frames capturados
frames = []
processing_queue = queue.Queue()
processing_started = False

def process_frames():
    global frames, processing_started
    detector = LicensePlateDetector(video_path=None, cascade_path='UKChars33_16x25_11W.xml')
    while True:
        if processing_started and frames:
            frame = frames.pop(0)
            processed_frame = detector.live(frame)  # Processa o frame na classe
            processing_queue.put(processed_frame)

# Inicia a thread de processamento
thread = threading.Thread(target=process_frames)
thread.start()

# Captura e salva frames
start_time = time.time()
while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Salva o frame no arquivo de vídeo
    out.write(frame)
    frames.append(frame)  # Adiciona o frame à lista

    # Inicia o processamento após 15 segundos
    if time.time() - start_time > 15:
        processing_started = True

    # Exibe frames processados na thread principal
    while not processing_queue.empty():
        processed_frame = processing_queue.get()
        cv2.imshow('Processed Frame', processed_frame)

    # Remoção de frames processados da lista
    if processing_started and len(frames) > 0:
        frames = frames[1:]  # Remove o primeiro frame processado

    # Condição de parada: pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
out.release()
cv2.destroyAllWindows()

