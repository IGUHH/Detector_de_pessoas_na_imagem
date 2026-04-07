import ultralytics
from ultralytics import YOLO
import cv2
import cv2_imshow

model = YOLO('yolov8n.pt') #first commit

imag = 'IMG_20260406_163434909.jpg'

img = cv2.imread(imag)
results = model.predict(source=imag, conf=0.5, classes=0)
annotated_frame = results[0].plot()

print(f"Exibindo resultados para: {imag}")
cv2_imshow(annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()