import cv2
import numpy as np

def order_points(pts):
    """Hàm chuẩn hóa thứ tự 4 góc: Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_qr_bounding_boxes_from_mask(mask):
    """
    Tìm QR code dựa trên mặt nạ trắng (Mask) đã được tiền xử lý.
    """
    qrs = []
    # Tìm các đường viền bao quanh các mảng trắng
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Bỏ qua các mảng trắng quá nhỏ (chắc chắn là nhiễu rác)
        if area < 1000:
            continue
            
        # Tìm hình chữ nhật nghiêng bọc vừa khít mảng trắng
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Kiểm tra tỷ lệ khung hình (Aspect Ratio)
        # Mã QR phải là hình vuông, tỷ lệ Aspect Ratio dao động quanh 1.0
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect_ratio = max(w, h) / min(w, h)
        
        # Chấp nhận hình vuông hoặc hình chữ nhật hơi méo (do chụp nghiêng)
        if aspect_ratio <= 1.8:
            
            # Chuẩn hóa thứ tự 4 góc theo yêu cầu của Grader
            ordered_box = order_points(box)
            
            qrs.append({
                "x0": float(ordered_box[0][0]), "y0": float(ordered_box[0][1]),
                "x1": float(ordered_box[1][0]), "y1": float(ordered_box[1][1]),
                "x2": float(ordered_box[2][0]), "y2": float(ordered_box[2][1]),
                "x3": float(ordered_box[3][0]), "y3": float(ordered_box[3][1]),
                "content": ""
            })
            
    return qrs