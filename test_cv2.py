import cv2
import numpy as np

img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.putText(img, "Test OpenCV GUI", (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
