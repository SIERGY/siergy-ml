# Model Machine Learning - Prediksi Masa Tanam Padi

Modul ini merupakan bagian dari *pipeline* *Machine Learning* untuk sistem informasi prediksi masa tanam komoditas pertanian. Skrip di dalam direktori ini bertanggung jawab untuk melatih ( *training* ) model klasifikasi menggunakan algoritma **Random Forest** untuk menentukan apakah kondisi tanah dan cuaca saat ini "Direkomendasikan" atau "Tidak Direkomendasikan" untuk penanaman padi.

## 📂 Struktur Direktori

Direktori ini fokus pada pemrosesan data dan pembuatan model. Output dari skrip ini akan digunakan oleh API backend (Flask).

* `main.py` : Skrip utama untuk memuat data, melatih model, mengevaluasi akurasi, dan mengekspor model.
* `Crop_recommendation.csv` : Dataset yang berisi parameter tanah dan iklim beserta label tanaman (sumber metrik: N, P, K, suhu, kelembapan, pH, dan curah hujan).
* `requirements.txt` : Daftar dependensi Python yang dibutuhkan untuk menjalankan proses *training*.

## 🛠️ Persyaratan Sistem

Sebelum menjalankan skrip *training*, pastikan *virtual environment* sudah aktif dan semua dependensi telah terinstal.

```bash
# Instalasi dependensi
pip install -r requirements.txt
```
## 🚀 Cara Menjalankan Skrip
Jalankan skrip `main.py` untuk memulai proses pelatihan model. Skrip ini akan secara otomatis memuat dataset, melakukan pembagian data, melatih model, mengevaluasi hasilnya, dan menyimpan model yang sudah dilatih.

```bash
python main.py
```
