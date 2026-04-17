import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_nms(qrs_list, distance_threshold=20):
    """
    Khử trùng lặp. 
    Nếu 2 mã QR có tâm nằm quá gần nhau, gộp chúng lại để tránh bị tính điểm FP (Thừa).
    """
    if len(qrs_list) == 0:
        return []
        
    keep = []
    # Lưu tâm và diện tích của từng box để so sánh
    boxes_info = []
    for qr in qrs_list:
        cx = (qr['x0'] + qr['x2']) / 2
        cy = (qr['y0'] + qr['y2']) / 2
        area = abs(qr['x2'] - qr['x0']) * abs(qr['y2'] - qr['y0'])
        boxes_info.append({'qr': qr, 'cx': cx, 'cy': cy, 'area': area})
        
    # Sắp xếp theo diện tích giảm dần (Ưu tiên giữ lại box to)
    boxes_info = sorted(boxes_info, key=lambda k: k['area'], reverse=True)
    
    for i in range(len(boxes_info)):
        box1 = boxes_info[i]
        should_keep = True
        
        # So sánh với các box đã được giữ lại
        for kept_box in keep:
            dist = np.sqrt((box1['cx'] - kept_box['cx'])**2 + (box1['cy'] - kept_box['cy'])**2)
            # Nếu tâm quá gần nhau (Cùng nằm trên 1 mã QR) -> Bỏ qua box này
            if dist < distance_threshold:
                should_keep = False
                break
                
        if should_keep:
            keep.append(box1)
            
    # Trả về danh sách đã lọc
    return [item['qr'] for item in keep]

def get_qr_bounding_boxes_from_mask(mask, min_solidity=0.9, min_area=350, aspect_ratio_threshold=2.2):
    raw_qrs = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        area = cv2.contourArea(c)
        
        if area < min_area:
            continue
            
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        
        
        if solidity < min_solidity:
            continue

        rect = cv2.minAreaRect(c)
        w, h = rect[1]
        if w < 18 or h < 18:
            continue
            
        aspect_ratio = max(w, h) / min(w, h)
        
        # 1. Mã QR chụp thẳng (Vuông vắn, cho phép hơi mẻ góc)
        is_normal_qr = (aspect_ratio <= aspect_ratio_threshold) and (solidity >= min_solidity)
        
        # 2. Mã QR chụp nghiêng cực độ (Hình chữ nhật dẹt/dài, BẮT BUỘC phải cực kỳ đặc)
        is_skewed_qr = (aspect_ratio_threshold < aspect_ratio <= 4.9) and (solidity >= 0.95)
        
        # Nếu thỏa mãn 1 trong 2 trường hợp thì lấy!
        if is_normal_qr or is_skewed_qr:
            
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            ordered_box = order_points(box)
            
            # Tính padding (5%)
            pad_w = int(w * 0.061)
            pad_h = int(h * 0.061)
            pad = max(pad_w, pad_h)
            
            raw_qrs.append({
                "x0": float(ordered_box[0][0]-pad), "y0": float(ordered_box[0][1]-pad),
                "x1": float(ordered_box[1][0]+pad), "y1": float(ordered_box[1][1]-pad),
                "x2": float(ordered_box[2][0]+pad), "y2": float(ordered_box[2][1]+pad),
                "x3": float(ordered_box[3][0]-pad), "y3": float(ordered_box[3][1]+pad),
                "content": ""
            })
            
    # Áp dụng bộ lọc khử trùng lặp trước khi trả về
    final_qrs = apply_nms(raw_qrs, distance_threshold=18)
    
    return final_qrs



