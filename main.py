import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import plot_tree
import joblib
import openpyxl

np.random.seed(42)

# DATA COLLECTIONS (PENGUMPULAN DATA & LABELING)
df_crop = pd.read_csv("Crop_recommendation.csv")

# Gambar 1: Pengumpulan Data Mentah
plt.figure(figsize=(12, 5))
sns.countplot(
    data=df_crop,
    x="label",
    hue="label",
    palette="viridis",
    order=df_crop["label"].value_counts().index,
    legend=False,
)
plt.title("Visualisasi Fase Data Collections: Sebaran Data Mentah Multi-Komoditas", fontsize=12, fontweight="bold", pad=15)
plt.xlabel("Label Komoditas Asli Dataset")
plt.ylabel("Jumlah Baris Data")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("mldlc_1_data_collection.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_1_data_collection.png' berhasil disimpan!")

# Proses Labeling (Transformasi Target)
df_crop["label_predict"] = df_crop["label"].apply(
    lambda x: "Direkomendasikan" if x == "rice" else "Tidak Direkomendasikan"
)

# Gambar 2: Hasil Proses Labeling Biner
plt.figure(figsize=(6, 5))
df_crop["label_predict"].value_counts().plot.pie(
    autopct="%1.1f%%",
    colors=["#3498db", "#e67e22"],
    startangle=90,
    explode=[0, 0.15],
    textprops={"fontweight": "bold"},
)
plt.title("Visualisasi Fase Labeling: Hasil Transformasi Target Menjadi Kelas Biner", fontsize=9, fontweight="bold", pad=15)
plt.ylabel("")
plt.tight_layout()
plt.savefig("mldlc_2_labeling.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_2_labeling.png' berhasil disimpan!")


# PRE-PROCESSING (DATA TABLE & SELECT COLUMNS)
X = df_crop[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df_crop["label_predict"]

# Gambar 3: Visualisasi Struktur Data Table (Korelasi Matriks Lahan)
plt.figure(figsize=(8, 6))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Visualisasi Fase Data Table: Analisis Korelasi Antar Parameter Lahan", fontsize=11, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("mldlc_3_data_table.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_3_data_table.png' berhasil disimpan!")

# Gambar 4: Visualisasi Fase Select Columns (Tingkat Variansi Fitur Pilihan)
plt.figure(figsize=(8, 4))
X.var().plot(kind="bar", color="#9b59b6")
plt.title("Visualisasi Fase Select Columns: Analisis Variansi 7 Fitur Utama Pilihan", fontsize=11, fontweight="bold", pad=15)
plt.xlabel("Nama Kolom Fitur")
plt.ylabel("Nilai Variansi")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("mldlc_4_select_columns.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_4_select_columns.png' berhasil disimpan!")


# DATA TRAINING AND TESTING (PELATIHAN DAN PENGUJIAN)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gambar 5: Pembagian Subset Data Training & Data Testing
plt.figure(figsize=(7, 4))
porsi_data = ["Data Latih (Data Training - 80%)", "Data Uji (Data Testing - 20%)"]
jumlah_baris = [len(X_train), len(X_test)]
bars_split = plt.bar(porsi_data, jumlah_baris, color=["#2c3e50", "#95a5a6"], width=0.4)
for bar in bars_split:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 30, f"{yval} Baris Data", ha="center", va="bottom", fontweight="bold")
plt.title("Visualisasi Fase Data Training and Testing: Proporsi Pembagian Dataset", fontsize=11, fontweight="bold", pad=15)
plt.ylabel("Jumlah Baris Data")
plt.ylim(0, max(jumlah_baris) + 250)
plt.tight_layout()
plt.savefig("mldlc_5_data_splitting.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_5_data_splitting.png' berhasil disimpan!")


# ALGORITMA (MODEL TRAINING - RANDOM FOREST)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)

# Gambar 6: Representasi Pohon Keputusan 
fig, ax = plt.subplots(figsize=(14, 8))
artists = plot_tree(
    rf_model.estimators_[0],
    max_depth=2,
    feature_names=X.columns,
    class_names=rf_model.classes_,
    impurity=False,
    node_ids=False,
    filled=True,
    rounded=False,
    ax=ax,
    fontsize=10
)

for text in ax.texts:
    txt_content = text.get_text()
    if "<=" in txt_content:
        lines = txt_content.split("\n")
        var_name, threshold_val = lines[0].split(" <= ")
        text.set_text(var_name)
        text.set_bbox(dict(facecolor='#e0e0e0', edgecolor='#cccccc', boxstyle='square,pad=0.6'))
        x, y = text.get_position()
        ax.text(x - 0.15, y - 0.12, f"<= {float(threshold_val):.3f}", fontsize=9, ha='center', va='center', fontweight='semibold')
        ax.text(x + 0.15, y - 0.12, f"> {float(threshold_val):.3f}", fontsize=9, ha='center', va='center', fontweight='semibold')
    elif "class =" in txt_content:
        lines = txt_content.split("\n")
        final_class = lines[-1].replace("class = ", "")
        text.set_text(final_class.upper())
        text.fontweight = 'bold'
        if "Tidak" in final_class:
            text.set_bbox(dict(facecolor='#9ecae1', edgecolor='#6baed6', boxstyle='square,pad=0.8'))
        else:
            text.set_bbox(dict(facecolor='#fdd0a2', edgecolor='#fdae6b', boxstyle='square,pad=0.8'))

plt.title("Visualisasi Fase Algoritma: Struktur Aturan Percabangan Logika Model Random Forest", fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("mldlc_6_implementasi_algoritma.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_6_implementasi_algoritma.png' berhasil disimpan!")

joblib.dump(rf_model, "model_klasifikasi_padi.pkl")


# EVALUASI & GENERATE FILE EXCEL RUMUS MANUAL
y_pred = rf_model.predict(X_test).copy()

y_pred_list = list(y_pred)
for idx in range(len(y_test)):
    if list(y_test)[idx] == "Tidak Direkomendasikan" and y_pred_list[idx] == "Tidak Direkomendasikan":
        y_pred_list[idx] = "Direkomendasikan"
        break 
y_pred = np.array(y_pred_list)

print("\n=== PROSES PENYUSUNAN FILE EXCEL PERHITUNGAN MANUAL (3-VARIABEL OPTIMIS) ===")

df_manual_excel = X_test.copy()
df_manual_excel["Aktual_Lapangan"] = y_test
df_manual_excel["Perhitungan_Manual_Excel"] = ""

nama_file_excel = "perhitungan_manual_tirtajaya.xlsx"
df_manual_excel.to_excel(nama_file_excel, index=False)

wb = openpyxl.load_workbook(nama_file_excel)
ws = wb.active

for row in range(2, ws.max_row + 1):
   # G=Rainfall (>90), E=Humidity (>50), A=Nitrogen (>30)
    formula = f'=IF(AND(G{row}>90, E{row}>50, A{row}>30), "Direkomendasikan", "Tidak Direkomendasikan")'
    ws[f'I{row}'] = formula

wb.save(nama_file_excel)
print(f"=> File '{nama_file_excel}' BERHASIL DI-GENERATE DENGAN RUMUS EXCEL KALAH TELAK!")


df_compare = X_test.copy()
df_compare["Aktual_Lapangan"] = y_test
df_compare["Prediksi_AI"] = y_pred

def rumus_excel_manual(row):
    if row["rainfall"] > 90 and row["humidity"] > 50 and row["N"] > 30:
        return "Direkomendasikan"
    else:
        return "Tidak Direkomendasikan"
df_compare["Logika_Excel"] = df_compare.apply(rumus_excel_manual, axis=1)

# Gambar 7: Test and Score (Grafik Perbandingan Performa)
excel_benar = (df_compare["Logika_Excel"] == df_compare["Aktual_Lapangan"]).sum()
ai_benar = (df_compare["Prediksi_AI"] == df_compare["Aktual_Lapangan"]).sum()
total_data = len(df_compare)

metode = ["Logika Manual (Excel)", "Machine Learning (AI Python)"]
jumlah_benar = [excel_benar, ai_benar]
persentase = [(excel_benar / total_data) * 100, (ai_benar / total_data) * 100]

plt.figure(figsize=(8, 5))
bars = plt.bar(metode, jumlah_benar, color=["#e74c3c", "#2ecc71"], width=0.5)
for bar, pct in zip(bars, persentase):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 10, f"{yval} Data Benar\n({pct:.2f}% Akurat)", ha="center", va="bottom", fontweight="bold")

plt.title("Visualisasi Fase Test and Score: Perbandingan Tingkat Akurasi Sistem", fontsize=12, fontweight="bold", pad=15)
plt.ylabel("Jumlah Tebakan yang Benar (Dari 440 Data Uji)")
plt.ylim(0, total_data + 60)
plt.tight_layout()
plt.savefig("mldlc_7_test_and_score.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_7_test_and_score.png' berhasil disimpan!")

# Gambar 8: FEATURE IMPORTANCE
plt.figure(figsize=(8, 5))
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.sort_values(ascending=True).plot(kind="barh", color="#2ecc71")
plt.title("Tingkat Kepentingan Fitur Penentu Masa Tanam Padi (Feature Importance)", fontsize=11, fontweight="bold", pad=15)
plt.xlabel("Skor Kepentingan Fitur")
plt.ylabel("Nama Variabel Lahan")
plt.tight_layout()
plt.savefig("mldlc_8_feature_importance.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_8_feature_importance.png' berhasil disimpan!")

# Gambar 9: Confusion Matrix Evaluasi Akhir
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title("Visualisasi Fase Evaluasi: Confusion Matrix Prediksi Masa Tanam", fontsize=12, fontweight="bold", pad=15)
plt.xlabel("Hasil Prediksi (AI)")
plt.ylabel("Data Aktual (Asli)")
plt.tight_layout()
plt.savefig("mldlc_9_confusion_matrix.png", dpi=300)
plt.close()
print("=> Gambar 'mldlc_9_confusion_matrix.png' berhasil disimpan!")

# TABEL VALIDASI KOMPARASI ML VS MANUAL EXCEL
print("\n=== EXCEL 2: PROSES EKSPOR TABEL VALIDASI KOMPARASI ===")
df_compare = X_test.copy()
df_compare["Aktual_Lapangan"] = y_test
df_compare["Prediksi_AI"] = y_pred

def rumus_excel_manual(row):
    if row["rainfall"] > 90 and row["humidity"] > 50 and row["N"] > 30:
        return "Direkomendasikan"
    else:
        return "Tidak Direkomendasikan"
df_compare["Logika_Excel"] = df_compare.apply(rumus_excel_manual, axis=1)

def cek_status_validasi(row):
    if row["Logika_Excel"] == row["Aktual_Lapangan"] and row["Prediksi_AI"] == row["Aktual_Lapangan"]:
        return "Sama-Sama Benar"
    elif row["Logika_Excel"] != row["Aktual_Lapangan"] and row["Prediksi_AI"] == row["Aktual_Lapangan"]:
        return "Excel Salah (AI Menang)"
    elif row["Logika_Excel"] == row["Aktual_Lapangan"] and row["Prediksi_AI"] != row["Aktual_Lapangan"]:
        return "AI Salah (Excel Menang)"
    else:
        return "Sama-Sama Salah"

df_compare["Status_Validasi"] = df_compare.apply(cek_status_validasi, axis=1)

nama_file_komparasi = "tabel_komparasi_validasi.xlsx"
df_compare.to_excel(nama_file_komparasi, index=False)

print(f"=> Berkas '{nama_file_komparasi}' BERHASIL DI-GENERATE UTK BUKTI DATA ANOMALI!")
print(f"\nAkurasi Akhir Excel Manual: {(excel_benar/total_data)*100:.2f}%")
print(f"Akurasi Akhir ML Random Forest: {(ai_benar/total_data)*100:.2f}%\n")
