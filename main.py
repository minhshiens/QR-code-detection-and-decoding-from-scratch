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
            

            # Bộ lọc 4 tầng để phát hiện đa dạng các loại mã QR khác nhau, từ rõ nét đến cực kỳ khó nhận diện:
            
            # TẦNG 1: Mã QR Tiêu chuẩn (Nét, vuông vắn, rõ ràng)
            qrs_fine = get_qr_bounding_boxes_from_mask(mask_fine, min_solidity=0.91, min_area=350, aspect_ratio_threshold=2.2)
            
            if len(qrs_fine) > 0:
                qrs = qrs_fine
                
            else:
                # TẦNG 2: Mã QR Khổng lồ / Đứt gãy li ti
                qrs_coarse = get_qr_bounding_boxes_from_mask(mask_coarse, min_solidity=0.81, min_area=9250, aspect_ratio_threshold=2.9)
                
                if len(qrs_coarse) > 0:
                    qrs = qrs_coarse
                    
                else:
                    # TẦNG 3: Mã QR Siêu nhỏ / Nghiêng xéo cực độ
                    qrs_extreme = get_qr_bounding_boxes_from_mask(mask_fine, min_solidity=0.82, min_area=120, aspect_ratio_threshold=4.5)
                    
                    if len(qrs_extreme) > 0:
                        qrs = qrs_extreme
                        
                    else:
                        # TẦNG 4: Mã QR bị rách, bị ngón tay che, bị dán tem đè
                        qrs_torn = get_qr_bounding_boxes_from_mask(mask_coarse, min_solidity=0.76, min_area=500, aspect_ratio_threshold=2.0)
                        qrs = qrs_torn
            
            # 2. Khử trùng lặp (Nhiều box chồng lên nhau do cùng nằm trên 1 mã QR) -> Giữ lại box to nhất
            verified_qrs = []
            for qr in qrs:
                if verify_qr(img, qr):
                    verified_qrs.append(qr)
                    
            result_dict["qrs"] = verified_qrs

            

        all_results.append(result_dict)


    # Dừng bấm giờ
    process_time = time.time() - start_time
    print(f" Tổng thời gian chạy: {process_time:.2f} giây")
    print(f" Tốc độ trung bình: {process_time/len(image_list):.4f} giây/ảnh")

    # 4. Ghi file output
    write_output_csv(all_results, "output.csv")

if __name__ == "__main__":
    main()