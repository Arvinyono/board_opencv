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

# Fungsi deteksi warna dalam range HSV
def detect_color(hsv_frame, color):
    if color == 'red':
        # Red memiliki dua range karena di ujung lingkaran HSV
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
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
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)

    # Cari kontur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_point = None

    if contours:
        # Ambil kontur terbesar
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 500:  # Hanya jika cukup besar
            x, y, w, h = cv2.boundingRect(contour)
            current_point = (x + w//2, y + h//2)
            cv2.circle(frame, current_point, 10, colors[color_names[current_color]], 2)

    # Gambar garis di canvas
    if prev_point and current_point:
        cv2.line(canvas, prev_point, current_point, colors[color_names[current_color]], 5)

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
        prev_point = None  # Reset posisi
    elif key == ord('s'):  # Simpan gambar
        filename = f"whiteboard_{np.random.randint(1000, 9999)}.png"
        cv2.imwrite(filename, canvas)
        print(f"Gambar disimpan sebagai {filename}")

# Tutup semua
cap.release()
cv2.destroyAllWindows()
