import cv2
import numpy as np


def find_finder_patterns(binary_img):
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    fps = [] 

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # NỚI LỎNG 1: Chấp nhận các góc QR nhỏ hơn (giảm từ 15 xuống 8 pixel)
        if w < 8 or h < 8: 
            continue
            
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # NỚI LỎNG 2: Không cần xấp xỉ chính xác 4 cạnh, các góc bị mờ tròn cũng chấp nhận
        if len(approx) >= 4 or len(approx) <= 8:
            child_idx = hierarchy[i][2] 
            
            # NỚI LỎNG 3: Chỉ cần lồng 2 lớp (1 vòng ngoài, 1 lõi trong) thay vì 3 lớp
            # Lý do: Ảnh mờ thường làm dính vòng viền trắng vào viền đen
            if child_idx != -1:
                area_outer = cv2.contourArea(contour)
                area_inner = cv2.contourArea(contours[child_idx])
                
                if area_inner > 0:
                    ratio = area_outer / area_inner
                    
                    # NỚI LỎNG 4: Tỷ lệ diện tích lỏng lẻo hơn rất nhiều (từ 1.1 đến 10.0)
                    if 1.1 <= ratio <= 10.0:
                        aspect_ratio = float(w) / h
                        if 0.3 <= aspect_ratio <= 3.0: # Chấp nhận ảnh bị nghiêng xéo
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                fps.append({"center": (cx, cy), "contour": contour})

    return fps

def get_qr_bounding_boxes(fps):
    """ Từ danh sách các ô vuông góc, gom nhóm và trả về 4 góc của QR code"""
    
    qrs = []

    # nếu không đủ 3 ô vuông góc thì không thể tạo thành QR code
    if len(fps) ==0:
        return qrs
    
    # Lấy ra tất cả các pixels của các viền ô vuông

    all_points = []
    for fp in fps:
        all_points.extend(fp["contour"])
    all_points = np.array(all_points)

    x, y, w, h = cv2.boundingRect(all_points)

    pad = int(w * 0.02)
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = w + 2*pad
    h = h + 2*pad


    qr_box = {
        "x0": x,
        "y0": y,
        "x1": x + w,
        "y1": y,
        "x2": x + w,
        "y2": y + h,
        "x3": x,
        "y3": y + h,
        "content": "" # sẽ được cập nhật sau
    }

    # Hiện tại chỉ có 1 QR code duy nhất, nếu có nhiều hơn thì cần phải phân nhóm các ô vuông góc lại với nhau

    qrs.append(qr_box)
    return qrs

    
