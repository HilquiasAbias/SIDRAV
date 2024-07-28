import cv2
import imutils
import numpy as np

def find_locations(frame):
  # Aplica o filtro bilateral para suavizar a imagem
  blurred_image = cv2.bilateralFilter(frame, 11, 17, 17)
  
  # Detecta bordas usando o detector de bordas Canny
  edged = cv2.Canny(blurred_image, 30, 200)
  
  # Encontra contornos na imagem binária
  keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(keypoints)
  
  # Ordena os contornos por área em ordem decrescente e seleciona os 10 maiores
  contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
  
  # Lista para armazenar as localizações de interesse
  locations = []
  
  for contour in contours:
    # Aproxima os contornos para formas poligonais
    approx = cv2.approxPolyDP(contour, 10, True)
    
    # Se o polígono tiver 4 vértices, é considerado uma localização
    if len(approx) == 4:
      locations.append(approx)
  
  return locations
