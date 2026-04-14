import cv2
import numpy as np

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gradX = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gradY = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    
    gradient = cv2.magnitude(gradX, gradY)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # ÁP DỤNG LỌC MỜ NHẸ ĐỂ GIẢM NHIỄU TRƯỚC KHI THRESHOLD
    blurred = cv2.blur(gradient, (4, 4))

    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    
    # THU NHỎ CHỔI QUÉT XUỐNG để không làm hỏng QR nhỏ
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Xói mòn và giãn nở để loại bỏ nhiễu nhỏ và làm mịn vùng phát hiện
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)
    
    return closed, gray