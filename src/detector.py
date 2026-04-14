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
    qrs = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # 1. HẠ NGƯỠNG DIỆN TÍCH TỪ 1000 XUỐNG 350 ĐỂ BẮT QR NHỎ
        if area < 350:
            continue
            
        # 2. KIỂM TRA SOLIDITY (ĐỘ ĐẶC)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        
        # QR Code bôi nhòe phải là một khối rất đặc (Solidity > 0.8)
        # Các bụi cây, hàng rào, đoạn văn bản sẽ lồi lõm và bị loại ở đây
        if solidity < 0.9:
            continue

        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        
        # Chiều dài/rộng tối thiểu của 1 QR code
        if w < 10 or h < 10:
            continue
            
        aspect_ratio = max(w, h) / min(w, h)
        
        # 3. SIẾT CHẶT TỶ LỆ VUÔNG (Giảm FP)
        if aspect_ratio <= 2.2:
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            ordered_box = order_points(box)
            
            # Thêm padding nhẹ 3 pixel để viền lấy trọn vẹn điểm IoU
            pad_w =int(w*0.09)
            pad_h =int(h*0.09)
            pad = max(pad_w, pad_h)

            # pad = 3 # Điều chỉnh padding nếu cần (tăng lên nếu QR code bị cắt, giảm nếu bị dính vào vật khác)
            qrs.append({
                "x0": float(ordered_box[0][0]-pad), "y0": float(ordered_box[0][1]-pad),
                "x1": float(ordered_box[1][0]+pad), "y1": float(ordered_box[1][1]-pad),
                "x2": float(ordered_box[2][0]+pad), "y2": float(ordered_box[2][1]+pad),
                "x3": float(ordered_box[3][0]-pad), "y3": float(ordered_box[3][1]+pad),
                "content": ""
            })
            
    return qrs