import os
import csv

def get_image_paths(csv_path):
    """ Đọc file CSV """

    base_dir = os.path.dirname(csv_path)

    if base_dir == "":
        base_dir = "."

    if not os.path.exists(csv_path):
        print(f"không tìm thấy file {csv_path}.")
        return []
    
    data_list = []
    with open(csv_path, "r", encoding="utf-8") as file:
        # Đọc tự động theo tên cột
        reader = csv.DictReader(file)

        for row in reader:
            img_id = row["image_id"].strip()
            img_path_relative = row["image_path"].strip()
            img_path = os.path.join(base_dir, img_path_relative)

            data_list.append({
                "image_id": img_id,
                "path": img_path
                })


    return data_list

def write_output_csv(results, output_filename="output.csv"):
    headers = [
        "image_id", "qr_index",
        "x0", "y0", "x1", "y1", 
        "x2", "y2", "x3", "y3",
        "content"
    ]
    # SỬ DỤNG QUOTING VÀ ESCAPECHAR ĐỂ ĐẢM BẢO TÍNH TOÀN VẸN CỦA DỮ LIỆU (TRÁNH LỖI KHI NỘI DUNG CÓ DẤU PHẨY HOẶC DẤU NGẠCH)
    with open(output_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, escapechar='\\')
        writer.writerow(headers)

        for res in results:
            img_id = res["image_id"]
            qrs = res["qrs"]

            # Với trường hợp không có QR code  
            if len(qrs) == 0:
                writer.writerow([img_id, "", "", "", "", "", "", "", "", "", ""])
            else:
                for idx, qr in enumerate(qrs):
                    # ÉP KIỂU SANG SỐ NGUYÊN (INT) ĐỂ FILE GỌN VÀ CHUẨN PIXEL
                    writer.writerow([
                        img_id,
                        idx,
                        int(round(qr["x0"])), int(round(qr["y0"])), 
                        int(round(qr["x1"])), int(round(qr["y1"])),
                        int(round(qr["x2"])), int(round(qr["y2"])), 
                        int(round(qr["x3"])), int(round(qr["y3"])),
                        qr.get("content", "")
                    ])

    print(f"✅ Kết quả đã được ghi vào {output_filename}.")