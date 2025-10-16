# whiteboard.py
import cv2
import numpy as np

# Warna pena (format BGR)
colors = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255)
}
color_names = list(colors.keys())
current_color = 0  # Mulai dari biru

# Posisi sebelumnya untuk menggambar garis
prev_point = None

# Canvas untuk menggambar (putih)
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Buffer untuk smoothing posisi
points_buffer = []
MAX_BUFFER = 5  # Rata-rata dari 5 frame terakhir

# Fungsi cek kebulatan objek
def is_circular(contour, threshold=0.6):
    area = cv2.contourArea(contour)
    if area < 500:
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    return circularity > threshold  # Semakin mendekati 1, semakin bulat

# Fungsi deteksi warna dalam range HSV
def detect_color(hsv_frame, color):
    if color == 'red':
        lower1 = np.array([0, 150, 100])
        upper1 = np.array([5, 255, 255])
        lower2 = np.array([175, 150, 100])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_frame, lower1, upper1)
        mask2 = cv2.inRange(hsv_frame, lower2, upper2)
        return mask1 + mask2
    elif color == 'blue':
        lower = np.array([94, 80, 2])
        upper = np.array([126, 255, 255])
        return cv2.inRange(hsv_frame, lower, upper)
    elif color == 'green':
        lower = np.array([25, 52, 72])
        upper = np.array([102, 255, 255])
        return cv2.inRange(hsv_frame, lower, upper)
    elif color == 'yellow':
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
        return cv2.inRange(hsv_frame, lower, upper)

# Ambil video dari webcam
cap = cv2.VideoCapture(0)

print("Petunjuk:")
print(" - Gerakkan ujung jari/pena berwarna di depan kamera")
print(" - Tekan 'c' untuk ganti warna pena")
print(" - Tekan 's' untuk simpan gambar")
print(" - Tekan 'q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror agar nyaman digunakan
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Deteksi warna terpilih
    mask = detect_color(hsv, color_names[current_color])
    
    # Bersihkan noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_point = None

    if contours:
        # Urutkan dari terbesar
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        found = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000 and is_circular(contour, 0.6):  # Filter ukuran & bentuk
                x, y, w, h = cv2.boundingRect(contour)
                raw_point = (x + w//2, y + h//2)

                # Smoothing: rata-rata beberapa posisi terakhir
                points_buffer.append(np.array(raw_point))
                if len(points_buffer) > MAX_BUFFER:
                    points_buffer.pop(0)
                avg_point = np.mean(points_buffer, axis=0).astype(int)
                current_point = tuple(avg_point)

                # Gambar lingkaran di objek asli
                cv2.circle(frame, current_point, 10, colors[color_names[current_color]], 2)
                found = True
                break
        
        if not found:
            current_point = None
            points_buffer.clear()  # Reset jika tidak ada objek valid
    else:
        current_point = None
        points_buffer.clear()

    # Gambar garis di canvas hanya jika ada titik sebelumnya DAN saat ini
    if prev_point is not None and current_point is not None:
        cv2.line(canvas, prev_point, current_point, colors[color_names[current_color]], 5)

    # Update posisi sebelumnya (hanya jika ada deteksi valid)
    prev_point = current_point

    # Tampilkan petunjuk warna
    cv2.putText(frame, f'Color: {color_names[current_color].upper()}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, colors[color_names[current_color]], 2)

    # Gabungkan frame dan canvas (untuk preview)
    combined = np.hstack((frame, canvas))

    cv2.imshow("Virtual Whiteboard [Left: Camera | Right: Canvas]", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):  # Ganti warna
        current_color = (current_color + 1) % len(color_names)
        prev_point = None  # Reset garis
        points_buffer.clear()
    elif key == ord('s'):  # Simpan gambar
        filename = f"whiteboard_{np.random.randint(1000, 9999)}.png"
        cv2.imwrite(filename, canvas)
        print(f"Gambar disimpan sebagai {filename}")

# Tutup semua
cap.release()
cv2.destroyAllWindows()
