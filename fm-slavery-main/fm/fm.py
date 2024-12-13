import cv2 as cv


stop_sign_cascade = cv.CascadeClassifier('cascade_stop_sign.xml')


def detect_stop_sign():
    cap = cv.VideoCapture(0)

    while True:
        i, img = cap.read()
        if not i:
            break


        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


        stop_signs = stop_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(stop_signs) > 0:
            print("stop sign at: ", stop_signs)


        for (x, y, w, h) in stop_signs:
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(img, 'Stop Sign', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        cv.imshow('Stop Sign Detection', img)


        if cv.waitKey(1) & 0xFF == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()


detect_stop_sign()
