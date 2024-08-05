import cv2 as opencv
import numpy as np

def preprocess_image(img_path):
  license_plate = opencv.imread(img_path)
  h, w, _ = license_plate.shape
  license_plate = opencv.resize(license_plate, (int(w * 4), int(h * 4)))

  # Realce da área da placa usando um kernel de afiação
  sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
  sharpened_license_plate = opencv.filter2D(license_plate, -1, sharpening_kernel)

  # Aumento do contraste usando CLAHE
  lab = opencv.cvtColor(sharpened_license_plate, opencv.COLOR_BGR2LAB)
  l, a, b = opencv.split(lab)
  clahe = opencv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
  limg = opencv.merge([clahe.apply(l), a, b])
  enhanced_license_plate = opencv.cvtColor(limg, opencv.COLOR_LAB2BGR)
    
  # Conversão para escala de cinza, desfoque gaussiano e limiarização adaptativa
  grayscale_license_plate = opencv.cvtColor(enhanced_license_plate, opencv.COLOR_BGR2GRAY)
  blurred_license_plate = opencv.GaussianBlur(grayscale_license_plate, (3, 3), 0)
  _, thresholded_license_plate = opencv.threshold(blurred_license_plate, 0, 255, opencv.THRESH_BINARY + opencv.THRESH_OTSU)

  # Operações morfológicas para limpar a imagem
  morph_kernel = opencv.getStructuringElement(opencv.MORPH_RECT, (3, 3))
  opened_license_plate = opencv.morphologyEx(thresholded_license_plate, opencv.MORPH_OPEN, morph_kernel, iterations=1)
  inverted_license_plate = opencv.bitwise_not(opened_license_plate)

  # Salva a imagem pré-processada com o mesmo nome substituindo imagem original
  opencv.imwrite(img_path, inverted_license_plate)
