import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

df_crop = pd.read_csv("Crop_recommendation.csv")

df_crop['label_predict'] = df_crop['label'].apply(
    lambda x: "Direkomendasikan" if x == "rice" else "Tidak Direkomendasikan"
)

X = df_crop[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = df_crop["label_predict"] 


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("=== TABEL EVALUASI UNTUK BAB 3 ===")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=rf_model.classes_, 
            yticklabels=rf_model.classes_)
plt.title("Confusion Matrix - Prediksi Masa Tanam Padi")
plt.xlabel("Hasil Prediksi (AI)")
plt.ylabel("Data Aktual (Asli)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("=> Gambar 'confusion_matrix.png' berhasil disimpan!")

plt.figure(figsize=(8, 5))
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.sort_values(ascending=True).plot(kind="barh", color="#2ecc71")
plt.title("Tingkat Kepentingan Fitur Penentu Masa Tanam Padi")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("=> Gambar 'feature_importance.png' berhasil disimpan!")

joblib.dump(rf_model, "model_klasifikasi_padi.pkl")
print("=> Model berhasil disimpan sebagai 'model_klasifikasi_padi.pkl'")
