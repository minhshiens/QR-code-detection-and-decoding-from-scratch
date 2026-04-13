import argparse
import cv2
import time
from src.utils import get_image_paths, write_output_csv
from src.preprocessor import preprocess_image
from src.detector import find_finder_patterns, get_qr_bounding_boxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Đường dẫn file CSV")
    args = parser.parse_args()

    # Bắt đầu bấm giờ (Theo tiêu chí Tốc độ của Giảng viên)
    start_time = time.time()
    
    image_list = get_image_paths(args.data)
    if not image_list:
        return

    all_results = []

    for item in image_list:
        img_id = item['image_id']
        img_path = item['path']
        
        # Khởi tạo kết quả mặc định (không tìm thấy QR)
        result_dict = {
            "image_id": img_id,
            "qrs": []
        }
        
        img = cv2.imread(img_path)
        if img is not None:
            # 1. Tiền xử lý
            binary_img, gray_img = preprocess_image(img)
            
            # 2. Tìm các ô vuông
            fps = find_finder_patterns(binary_img)
            
            # 3. Tính toán 4 góc tọa độ
            qrs = get_qr_bounding_boxes(fps)
            result_dict["qrs"] = qrs
            
        all_results.append(result_dict)

    # Dừng bấm giờ
    process_time = time.time() - start_time
    print(f"⏱️ Tổng thời gian chạy: {process_time:.2f} giây")
    print(f"⚡ Tốc độ trung bình: {process_time/len(image_list):.4f} giây/ảnh")

    # 4. Ghi file output
    write_output_csv(all_results, "output.csv")

if __name__ == "__main__":
    main()