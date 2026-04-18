import cv2
import csv
import os
import numpy as np

def get_boxes_from_csv(csv_path, target_img_id):
    """Trích xuất tọa độ của 1 bức ảnh cụ thể từ file CSV"""
    boxes = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['image_id'] == target_img_id and row.get('x0'):
                try:
                    pts = np.array([
                        [float(row['x0']), float(row['y0'])],
                        [float(row['x1']), float(row['y1'])],
                        [float(row['x2']), float(row['y2'])],
                        [float(row['x3']), float(row['y3'])]
                    ], np.int32)
                    boxes.append(pts)
                except ValueError:
                    pass
    return boxes


imgpath = "lotsimage001_jpg.rf.750a41e5352a6b9b590aad4063b839c0"

def main():
    
    TARGET_IMG_ID = imgpath 
    
    # CHỈ ĐƯỜNG DẪN TỚI FILE ẢNH ĐÓ TRÊN MÁY BẠN
    IMG_PATH = f"train/{TARGET_IMG_ID}.jpg" 
    
    img = cv2.imread(IMG_PATH)
    if img is None:
        print("Không tìm thấy ảnh!")
        return

    # Lấy tọa độ từ 2 file
    gt_boxes = get_boxes_from_csv("output_train.csv", TARGET_IMG_ID)
    pred_boxes = get_boxes_from_csv("output.csv", TARGET_IMG_ID)

    # 1. VẼ ĐÁP ÁN (GROUND TRUTH) - MÀU XANH LÁ (Độ dày 2px)
    for box in gt_boxes:
        box = box.reshape((-1, 1, 2))
        cv2.polylines(img, [box], True, (0, 255, 0), 2)
        cv2.putText(img, "Dap An", tuple(box[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. VẼ DỰ ĐOÁN (PREDICTION) - MÀU ĐỎ (Độ dày 4px)
    for box in pred_boxes:
        box = box.reshape((-1, 1, 2))
        cv2.polylines(img, [box], True, (0, 0, 255), 4)
        cv2.putText(img, "Du Doan", tuple(box[3][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Hiển thị
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * 800 / h), 800))
    cv2.imshow("So Sanh IoU", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()