from paddleocr import PaddleOCR

def get_license_plate_text(img_path):
  ocr = PaddleOCR(use_angle_cls=True, lang='pt')
  result = ocr.ocr(img_path, cls=True)

  if result is None or len(result) == 0:
    return None
  else:
    text, confidence = result[0][0][1]
    return text
