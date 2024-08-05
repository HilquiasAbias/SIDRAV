import cv2
import threading
import time
import queue
import os
import shutil

from license_plate_detection.license_plate_detection import LicensePlateDetector

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

