import cv2
import os

cap = cv2.VideoCapture(0)

start = False
count = 200
num_samples = 300
IMG_CLASS_PATH = "Chinmay Recognition/image_Data/main"

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue

    if count == num_samples:
        break

    cv2.rectangle(frame, (150, 150), (500, 500), (255, 255, 255), 2)

    if start:
        roi = frame[150:500, 150:500]
        save_path = os.path.join(IMG_CLASS_PATH, '{}.jpg'.format(count + 1))
        cv2.imwrite(save_path, roi)
        count += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    k = cv2.waitKey(25)
    if k == ord('a'):
        start = not start

    if k == ord('q'):
        break

print("\n{} image(s) saved to {}".format(count, IMG_CLASS_PATH))
cap.release()
cv2.destroyAllWindows()