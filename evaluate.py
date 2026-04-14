import csv
import os
from shapely.geometry import Polygon

def load_data(csv_path):
    """
    Đọc file CSV và nhóm các khung bounding box (Polygon) theo từng image_id.
    """
    data = {}
    if not os.path.exists(csv_path):
        print(f"❌ Lỗi: Không tìm thấy file {csv_path}")
        return {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row['image_id']
            if img_id not in data:
                data[img_id] = []
            
            # Nếu có tọa độ thì mới tạo Polygon
            if row.get('x0') and row['x0'].strip() != "":
                try:
                    # Tạo tứ giác từ 4 điểm (x, y)
                    poly = Polygon([
                        (float(row['x0']), float(row['y0'])),
                        (float(row['x1']), float(row['y1'])),
                        (float(row['x2']), float(row['y2'])),
                        (float(row['x3']), float(row['y3']))
                    ])
                    # Làm chuẩn hóa tứ giác (nếu tự cắt chéo)
                    if not poly.is_valid:
                        poly = poly.buffer(0) 
                        
                    data[img_id].append(poly)
                except ValueError:
                    continue
    return data

def calculate_iou(poly1, poly2):
    """Tính tỷ lệ IoU giữa 2 hình đa giác"""
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception:
        return 0.0

def evaluate(pred_file="output.csv", gt_file="ground_truth.csv", iou_threshold=0.5):
    """
    Thuật toán Greedy IoU Matching theo đúng đặc tả của giảng viên.
    """
    print("⏳ Đang tính toán điểm số...")
    preds = load_data(pred_file)
    gts = load_data(gt_file)

    if not preds or not gts:
        print("Không có dữ liệu để chấm điểm.")
        return

    # Lấy danh sách tất cả các ảnh có trong 2 file
    all_images = set(preds.keys()).union(set(gts.keys()))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_id in all_images:
        pred_polys = preds.get(img_id, [])
        gt_polys = gts.get(img_id, [])

        matched_gt_indices = set() # Lưu vết các Ground Truth đã được ghép cặp

        # Với mỗi bounding box dự đoán
        for p_poly in pred_polys:
            best_iou = 0.0
            best_gt_idx = -1

            # Tìm Ground Truth có IoU cao nhất và chưa được ghép cặp
            for idx, g_poly in enumerate(gt_polys):
                if idx in matched_gt_indices:
                    continue
                
                iou = calculate_iou(p_poly, g_poly)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # Áp dụng ngưỡng IoU >= 0.5
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt_indices.add(best_gt_idx) # Đánh dấu GT này đã bị lấy
            else:
                total_fp += 1 # Đoán ra khung nhưng bị trượt, hoặc đoán thừa

        # Các Ground Truth chưa được ghép cặp nào thì tính là Bỏ sót (False Negative)
        total_fn += (len(gt_polys) - len(matched_gt_indices))

    # ===== TÍNH TOÁN METRICS CỐT LÕI =====
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # In kết quả ra màn hình
    print("\n" + "="*40)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*40)
    print(f"Tổng số ảnh đánh giá  : {len(all_images)}")
    print(f"True Positives (Đúng): {total_tp}")
    print(f"False Positives(Thừa): {total_fp}")
    print(f"False Negatives(Thiếu):{total_fn}")
    print("-" * 40)
    print(f"🎯 Precision (Độ chuẩn) : {precision:.4f}")
    print(f"🎯 Recall    (Độ phủ)   : {recall:.4f}")
    print(f"⭐ F1 SCORE             : {f1_score:.4f} ⭐")
    print("="*40)

if __name__ == "__main__":
    evaluate(pred_file="output.csv", gt_file="output_valid.csv")