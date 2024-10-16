import streamlit as st  
import numpy as np 
import cv2  
import torch 
import pathlib  
import logging  

# Mengatur jalur file untuk sistem operasi Windows
pathlib.PosixPath = pathlib.WindowsPath

# Konfigurasi logging
logging.basicConfig(level=logging.DEBUG)

# Memuat model YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')

# Label kelas untuk klasifikasi sampah
class_label = {
    0: "anorganik",
    1: "beracun",
    2: "kertas",
    3: "organik",
    4: "residu"
}

# Fungsi untuk menggambar kotak pembatas pada gambar
def draw_bounding_boxes(pred_tensor, result):
    font = cv2.FONT_HERSHEY_SIMPLEX  # Menetapkan font teks
    fontScale = 0.7  # Menetapkan skala font
    size_of_tensor = list(pred_tensor.size())  # Mendapatkan ukuran tensor prediksi
    rows = size_of_tensor[0]  # Mendapatkan jumlah baris dalam tensor
    for i in range(0, rows):  # Loop melalui setiap baris dalam tensor
        class_index = int(pred_tensor[i, 5].item())  # Mendapatkan indeks kelas prediksi
        # Menetapkan warna berdasarkan indeks kelas
        if class_index == 0:  # anorganik
            color = (0, 165, 255)  # Orange
        elif class_index == 1:  # beracun
            color = (0, 0, 255)  # Merah
        elif class_index == 2:  # kertas
            color = (255, 0, 0)  # Biru
        elif class_index == 3:  # organik
            color = (0, 255, 0)  # Hijau
        elif class_index == 4:  # residu
            color = (203, 192, 255)  # Pink

        # Menggambar kotak pembatas berdasarkan koordinat prediksi
        cv2.rectangle(result, (int(pred_tensor[i, 0].item()), int(pred_tensor[i, 1].item())),
                      (int(pred_tensor[i, 2].item()), int(pred_tensor[i, 3].item())), color, 2)

        # Menambahkan teks label dan skor ke gambar
        text = class_label[class_index] + " " + str(round(pred_tensor[i, 4].item(), 2))
        result = cv2.putText(result, text, (int(pred_tensor[i, 0].item()) + 5, int(pred_tensor[i, 1].item())),
                            font, fontScale, color, 2)

    return result

# Set judul 
st.title("PENDETEKSI JENIS SAMPAH \U0001F5D1")

# Navigasi sidebar 
tabs = ["Unggah Gambar", "Ambil Gambar", "Kamera Realtime"]
selected_tab = st.sidebar.radio("Navigasi", tabs)

# Tab 1: Unggah Gambar
if selected_tab == "Unggah Gambar":
    # Unggah gambar
    test_image = st.file_uploader('Unggah Gambar', type=['jpg', 'png', 'jpeg'])
    if test_image is not None:
        # Mengonversi gambar yang diunggah ke format OpenCV
        file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        test_image_decoded = cv2.imdecode(file_bytes, 1)

        # Menampilkan gambar yang diunggah
        st.image(test_image_decoded, channels="BGR", caption='Gambar yang Diunggah')

        # Membuat prediksi pada gambar yang diunggah
        prediction = model(test_image_decoded)
        result_img = draw_bounding_boxes(prediction.xyxy[0], test_image_decoded)
        st.image(result_img, channels="BGR", caption='Gambar yang Diprediksi')

# Tab 2: Ambil Gambar
elif selected_tab == "Ambil Gambar":
    # Ambil gambar dari kamera
    img_camera = st.camera_input("Ambil Gambar")
    if img_camera is not None:
        # Mengonversi gambar yang diambil ke format OpenCV
        file_bytes = np.asarray(bytearray(img_camera.read()), dtype=np.uint8)
        test_image_decoded = cv2.imdecode(file_bytes, 1)

        # Membuat prediksi pada gambar yang diambil
        prediction = model(test_image_decoded)
        result_img = draw_bounding_boxes(prediction.xyxy[0], test_image_decoded)
        st.image(result_img, channels="BGR", caption='Gambar yang Diprediksi')

# Tab 3: Kamera Realtime
elif selected_tab == "Kamera Realtime":
    stframe = st.empty()

    # Membuka koneksi ke webcam
    vid = cv2.VideoCapture(0)

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        # Mengonversi frame ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Membuat prediksi pada frame
        prediction = model(frame_rgb)

        # Menggambar kotak pembatas pada frame
        result_frame = draw_bounding_boxes(prediction.xyxy[0], frame_rgb)

        # Menampilkan frame dengan deteksi
        stframe.image(result_frame, channels="RGB", use_column_width=True)

    # Menutup koneksi ke webcam
    vid.release()
