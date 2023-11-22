import cv2

cap = cv2.VideoCapture(0)  # Try changing the index if 0 doesn't work

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Failed to grab frame")
        break

cap.release()
cv2.destroyAllWindows()