import numpy as np
import cv2

def apply_rotation(image, degrees):
  # Calcula o centro da imagem
  center = tuple(np.array(image.shape[1::-1]) / 2)

  # Constrói a matriz de rotação
  rotation_matrix = cv2.getRotationMatrix2D(center, degrees, 1.0)

  # Aplica a rotação na imagem
  rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
  return rotated_image

def estimate_skew_angle(image):
  # Garante que a imagem esteja em escala de cinza
  if len(image.shape) == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    gray_image = image

  # Aplica o filtro de desfoque mediano
  blurred_image = cv2.medianBlur(gray_image, 3)

  # Detecta as bordas na imagem
  edges = cv2.Canny(blurred_image, 30, 100, apertureSize=3, L2gradient=True)

  # Detecta linhas usando a Transformada de Hough
  height, width = edges.shape
  detected_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=width / 4.0, maxLineGap=height / 4.0)

  if detected_lines is None:
    return 0.0

  angle_sum = 0.0
  num_lines = 0

  # Calcula o ângulo médio das linhas detectadas
  for line in detected_lines:
    for x1, y1, x2, y2 in line:
      theta = np.arctan2(y2 - y1, x2 - x1)
      if np.abs(theta) <= np.radians(30):  # Filtra ângulos extremos
        angle_sum += theta
        num_lines += 1

  if num_lines == 0:
    return 0.0

  # Calcula o ângulo médio em graus
  average_angle = (angle_sum / num_lines) * (180 / np.pi)
  return average_angle

def correct_skew(image):
  # Estima o ângulo de inclinação da imagem
  skew_angle = estimate_skew_angle(image)

  # Aplica a rotação para corrigir a inclinação
  return apply_rotation(image, skew_angle)
