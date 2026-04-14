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
    kernel_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    mask_fine = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_fine)
    
    # Xói mòn và giãn nở để loại bỏ nhiễu nhỏ và làm mịn vùng phát hiện
    mask_fine = cv2.erode(mask_fine, None, iterations=2)
    mask_fine = cv2.dilate(mask_fine, None, iterations=2)

    kernel_coarse = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_coarse = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_coarse)
    
    return mask_fine, mask_coarse, gray