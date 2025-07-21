import cv2
import easyocr
import time

# —————— 1. Load Haar Cascade for license plates ——————
cascade_path = "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():
    raise IOError(f"Cannot load cascade at {cascade_path}")

# —————— 2. Init EasyOCR reader ——————
reader = easyocr.Reader(['en'])

# —————— 3. Open your webcam ——————
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(2)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

# —————— 4. Main loop ——————
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,20)
    )

    for (x, y, w, h) in plates:
        plate_img = frame[y:y+h, x:x+w]
        if plate_img.size == 0:
            continue

        # — OCR on plate region —
        ocr_results = reader.readtext(plate_img)
        text = ocr_results[0][1] if ocr_results else "No Text"

        # — Draw box + text —
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    cv2.imshow("Plate → OCR", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
