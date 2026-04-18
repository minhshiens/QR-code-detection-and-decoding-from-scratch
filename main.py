import argparse
import cv2
import time
from src.utils import get_image_paths, write_output_csv
from src.preprocessor import preprocess_image
from src.detector import get_qr_bounding_boxes_from_mask, verify_qr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="Đường dẫn file CSV")
    args = parser.parse_args()

    # Bấm giờ bắt đầu
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
            # 1. Tiền xử lý (Trả ra mask đã bôi nhòe)
            mask_fine, mask_coarse, gray = preprocess_image(img)
            
            
            # 2. Lấy tọa độ trực tiếp từ mask
            qrs_fine = get_qr_bounding_boxes_from_mask(mask_fine, min_solidity=0.91, min_area=350, aspect_ratio_threshold=2.2)

            if len(qrs_fine) == 0:
                qrs_coarse = get_qr_bounding_boxes_from_mask(mask_coarse, min_solidity=0.81, min_area=9250, aspect_ratio_threshold=2.9)
                qrs = qrs_coarse
            else:
                qrs = qrs_fine

            verified_qrs = []
            for qr in qrs:
                if verify_qr(img, qr):
                    verified_qrs.append(qr)
                    
            result_dict["qrs"] = verified_qrs


            # result_dict["qrs"] = qrs
            

        all_results.append(result_dict)


    # Dừng bấm giờ
    process_time = time.time() - start_time
    print(f" Tổng thời gian chạy: {process_time:.2f} giây")
    print(f" Tốc độ trung bình: {process_time/len(image_list):.4f} giây/ảnh")

    # 4. Ghi file output
    write_output_csv(all_results, "output.csv")

if __name__ == "__main__":
    main()