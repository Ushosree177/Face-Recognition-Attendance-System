import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) == 27:   # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()