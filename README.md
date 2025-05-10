# decision_trees
Bu kod, karar ağacı tabanlı regresyon modellerinin tahmin performansını analiz etmek ve hangi modelin daha iyi çalıştığını belirlemek için kullanılır.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

drive.mount('/content/drive')

# 1. Adım: Veriyi yükleme
database_path = input("Lütfen veritabanı dosyasının yolunu girin: ")
df = pd.read_csv(database_path)

print(df.info())
print(df.describe().T)

# 2. Adım: Bağımlı değişkeni seçme
target_column = input("Bağımlı değişkenin adını girin: ")

# 3. Adım: Sayısal ve kategorik sütunları ayırma
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(exclude=np.number).columns

# 4. Adım: Null değerleri doldurma
numerical_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# 5. Adım: Kategorik değişkenleri one-hot encoding ile dönüştürme
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 6. Adım: Bağımsız ve bağımlı değişkenleri ayırma
X = df_encoded.drop(target_column, axis=1)
y = df_encoded[target_column]

# 7. Adım: Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Adım: Ölçeklendirme işlemi (Karar ağaçları için genellikle gerekli değil)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Adım: Modelleri tanımlama ve karşılaştırma
models = {
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=5),
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, max_depth=5),
    'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, max_depth=5)
}

results = {}

for name, model in models.items():
    # Model eğitme ve değerlendirme
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    results[name] = {"R²": r2, "MSE": mse, "RMSE": rmse}

# Sonuçları DataFrame formatında yazdırma
results_df = pd.DataFrame(results).T
print("\nKarar Ağaçları Model Performans Sonuçları:")
print(results_df.to_string(float_format="%.4f"))

# 10. Adım: Sonuçları görselleştirme

# R² skorları çubuk grafiği
plt.figure(figsize=(10, 5))
r2_scores = {name: values["R²"] for name, values in results.items()}
plt.bar(r2_scores.keys(), r2_scores.values(), color=['blue', 'green', 'red'])
plt.xticks(rotation=45, ha='right')
plt.xlabel("Modeller")
plt.ylabel("R² Skoru")
plt.title("Karar Ağaçları Modellerinin Performansı")
plt.show()

# Gerçek vs Tahmin Edilen Değerler dağılım grafiği
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs Tahmin Edilen Değerler")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')  # Doğru tahmin çizgisi
plt.show()

# Hata Dağılımı histogramı
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color="orange")
plt.xlabel("Hata (Residuals)")
plt.ylabel("Frekans")
plt.title("Karar Ağacı Model Hata Dağılımı")
plt.show()

# Confusion Matrix için tahmin ve gerçek değerleri kategorilere ayırma
thresholds = np.percentile(y_test, [25, 50, 75])  # Veri yüzdelik dilimlerine göre aralıklar oluştur
def categorize(value, thresholds):
    if value <= thresholds[0]:
        return "Low"
    elif value <= thresholds[1]:
        return "Medium"
    elif value <= thresholds[2]:
        return "High"
    else:
        return "Very High"

y_test_cat = np.array([categorize(val, thresholds) for val in y_test])
y_pred_cat = np.array([categorize(val, thresholds) for val in y_pred])

# Confusion matrix oluşturma ve görselleştirme
cm = confusion_matrix(y_test_cat, y_pred_cat, labels=["Low", "Medium", "High", "Very High"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High", "Very High"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Regresyon Tahminlerinden Dönüştürülen)")
plt.show()
