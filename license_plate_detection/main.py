import cv2
import os
import shutil
import argparse

from functions.skew_correction import correct_skew
from functions.find_locations import find_locations
from functions.enhance_lines import enhance_lines

class LicensePlateDetector:
  def __init__(self, video_path, cascade_path, debug=False):
    if video_path is not None:
      self.video = cv2.VideoCapture(video_path)
    self.characters_cascade = cv2.CascadeClassifier(cascade_path)
    self.debug = debug
    self.trackers = {}

    if os.path.exists('plates'):
      shutil.rmtree('plates')
    os.makedirs('plates')

  def process_frame(self, img):
    img = cv2.resize(img, (620, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    locations = find_locations(frame=gray)

    for location in locations:
      (x, y, w, h) = cv2.boundingRect(location)
      plate = img[y:y+h, x:x+w]
      if w > h: #  and w >= 1.5 * h
        if self.debug:
          cv2.imshow('roi', plate)
          
        gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        plate_resized = cv2.resize(gray_plate, (int(w * 4), int(h * 4)))
        plate = enhance_lines(plate_resized)
        chars = self.characters_cascade.detectMultiScale(plate, 1.1, 5)

        roi_area = w * h
        frame_area = img.shape[1] * img.shape[0]
        if 0.001 < roi_area / frame_area < 0.01:
          if self.debug:
            cv2.imshow('roi 2', plate)
          # gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
          # plate_resized = cv2.resize(gray_plate, (int(w * 4), int(h * 4)))
          # plate = enhance_lines(plate_resized)
          # chars = self.characters_cascade.detectMultiScale(plate, 1.1, 5)

          # if self.debug:
          #   cv2.imshow('possible license plate', plate)

          if len(chars) < 1:
            try:
              plate = correct_skew(image=plate)
              chars = self.characters_cascade.detectMultiScale(plate, 1.1, 5)
            except Exception as e:
              if self.debug:
                print('Error during skew correction:', e)
              continue

          if len(chars) > 0:
            found = False
            for tracker_id, tracker in self.trackers.items():
              (_, bbox) = tracker['instance'].update(img)
              (x2, y2, w2, h2) = [int(v) for v in bbox]
              if (
                (abs(x - x2) <= 0.6 * w2 and abs(y - y2) <= 0.6 * h2) or
                (abs(x - x2) <= 0.6 * w2 and abs((y + h) - (y2 + h2)) <= 0.6 * h2) or
                (abs((x + w) - (x2 + w2)) <= 0.6 * w2 and abs(y - y2) <= 0.6 * h2) or
                (abs((x + w) - (x2 + w2)) <= 0.6 * w2 and abs((y + h) - (y2 + h2)) <= 0.6 * h2)
              ):
                found = True
                break

            if not found:
              tracker = cv2.TrackerCSRT_create()
              tracker.init(img, (x, y, w, h))
              self.trackers[len(self.trackers)] = {'instance': tracker, 'count': 1}

              if self.debug:
                cv2.imshow('plate_' + str(len(self.trackers) - 1), img[y-10:y+h+10, x-10:x+w+10])

    self.update_trackers(img)

    if self.debug:
      cv2.imshow('Video', img)

  def update_trackers(self, img):
    for tracker_id, tracker in list(self.trackers.items()):
      (_, bbox) = tracker['instance'].update(img)
      (x, y, w, h) = [int(v) for v in bbox]

      if x == 0 and y == 0 and w == 0 and h == 0:
        continue

      tracker_count = tracker['count']

      if tracker_count >= 6:
        continue

      path = os.path.join('plates', str(tracker_id))
      os.makedirs(path, exist_ok=True)

      if tracker_count == 1:
        cv2.imwrite(os.path.join(path, 'frame.jpg'), img)

      plate_path = os.path.join(path, 'plate' + str(tracker_count) + '.jpg')

      try:
        cv2.imwrite(plate_path, img[y-10:y+h+10, x-10:x+w+10])
      except Exception as e:
        if self.debug:
          print('Error saving image:', e)

      self.trackers[tracker_id]['count'] += 1

      if self.debug:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

  def run(self):
    while True:
      ret, img = self.video.read()
      if img is None:
        break

      self.process_frame(img)

      if cv2.waitKey(33) == 27 and self.debug:
        break

    self.video.release()
    cv2.destroyAllWindows()

  def live(self, frame):
    self.process_frame(frame)
    return frame


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="License Plate Detector")
  parser.add_argument("--video_path", type=str, default='tests/videos/video7.mp4', help="Caminho para o vídeo")
  parser.add_argument("--chars_cascade_path", type=str, default='UKChars33_16x25_11W.xml', help="Caminho para o arquivo de cascade de caracteres")
  parser.add_argument("--debug", type=bool, default=False, help="Modo de debug")

  args = parser.parse_args()

  detector = LicensePlateDetector(video_path=args.video_path, cascade_path=args.chars_cascade_path, debug=args.debug)
  detector.run()
