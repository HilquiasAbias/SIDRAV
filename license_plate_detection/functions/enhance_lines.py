import numpy as np
import cv2

def enhance_lines(image):
    # Detecta bordas usando o detector de bordas Canny
    bordas = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Detecta linhas usando a Transformada de Hough
    linhas_detectadas = cv2.HoughLines(bordas, 1, np.pi / 180, 200)
    
    if linhas_detectadas is not None:
        for rho, theta in linhas_detectadas[:, 0]:
            # Calcula os pontos de início e fim para desenhar a linha
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            x0 = cos_theta * rho
            y0 = sin_theta * rho
            x1 = int(x0 + 1000 * (-sin_theta))
            y1 = int(y0 + 1000 * (cos_theta))
            x2 = int(x0 - 1000 * (-sin_theta))
            y2 = int(y0 - 1000 * (cos_theta))
            
            # Desenha a linha detectada na imagem
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 6)
        
        # Aplica um filtro bilateral para melhorar a imagem
        imagem_filtrada = cv2.bilateralFilter(image, 11, 17, 17)
        return imagem_filtrada

    return image
