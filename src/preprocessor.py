import cv2

def preprocess_image(img):
    """
    Tiền xử lý ảnh: Chuyển ảnh màu sang ảnh nhị phân đen trắng
    để chuẩn bị cho việc tìm các hình vuông của QR Code.
    """
    # 1. Chuyển ảnh màu sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Làm mờ nhẹ (Gaussian Blur) để khử nhiễu (các hạt sạn nhỏ trong ảnh)
    # Kích thước kernel 5x5
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Kỹ thuật: Adaptive Threshold
    # Đừng dùng Threshold cứng (Global), vì nếu ảnh bị đổ bóng một nửa, 
    # Threshold cứng sẽ làm đen thui phần bị bóng râm.
    # Adaptive Threshold sẽ tính toán độ sáng dựa trên các vùng nhỏ (21x21 pixel).
    # THRESH_BINARY_INV: Đảo ngược màu -> Viền đen của QR sẽ biến thành màu TRẮNG 
    # (thuận lợi cho thuật toán tìm biên).
    binary = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 5
    )
    
    return binary, gray