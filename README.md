# Karar Ağacı Tabanlı Regresyon Modeli Analizi

Bu proje, çeşitli karar ağacı tabanlı regresyon modellerinin (Karar Ağacı Regresyonu, Rastgele Orman Regresyonu ve Gradyan Artırma Regresyonu) tahmin performansını analiz etmek ve karşılaştırmak için Python tabanlı bir araç sunar. Kullanıcı tarafından sağlanan bir veri kümesi üzerinde çalışır, veriyi ön işler, modelleri eğitir, değerlendirir ve sonuçları görselleştirir.

## Özellikler

*   Farklı karar ağacı tabanlı regresyon modellerinin performans karşılaştırması.
*   Veri ön işleme adımları:
    *   Eksik değerlerin doldurulması (sayısal sütunlar için ortalama, kategorik sütunlar için en sık kullanılan değer).
    *   Kategorik değişkenlerin one-hot encoding ile dönüştürülmesi.
*   Model performans metrikleri: R², Ortalama Kare Hata (MSE), Kök Ortalama Kare Hata (RMSE).
*   Sonuçların görselleştirilmesi:
    *   Modellerin R² skorlarını gösteren çubuk grafik.
    *   Gerçek ve tahmin edilen değerlerin dağılım grafiği.
    *   Model hata dağılımını gösteren histogram.
    *   Regresyon tahminlerinden dönüştürülmüş bir confusion matrix.

## Kurulum

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Python Kurulumu:**
    Makinenizde Python 3.x sürümünün kurulu olduğundan emin olun. [Python resmi web sitesinden](https://www.python.org/downloads/) indirebilirsiniz.

2.  **Gerekli Kütüphaneler:**
    Projenin çalışması için aşağıdaki Python kütüphanelerine ihtiyaç vardır. Bu kütüphaneleri pip kullanarak kurabilirsiniz:

    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

    Eğer Google Colab ortamında çalıştırıyorsanız, bu kütüphanelerin çoğu zaten kurulu olacaktır. `google.colab` kütüphanesi sadece Colab ortamında gereklidir ve yerel kurulumlarda bu satır atlanabilir veya yorum satırı haline getirilebilir.

## Kullanım

Proje betiğini çalıştırmak için aşağıdaki adımları izleyin:

1.  **Veri Seti Hazırlığı:**
    Analiz etmek istediğiniz veri setini CSV formatında hazırlayın.

2.  **Betiği Çalıştırma:**
    Python betiğini (örneğin `decision_tree_analysis.py` olarak kaydedilmişse) bir terminal veya komut istemcisi üzerinden çalıştırın:

    ```bash
    python decision_tree_analysis.py
    ```

    Alternatif olarak, kodu bir Jupyter Notebook (`.ipynb`) dosyasına kopyalayıp hücreleri sırayla çalıştırabilirsiniz.

3.  **Girdi Sağlama:**
    Betiği çalıştırdığınızda, sizden aşağıdaki bilgileri girmeniz istenecektir:
    *   **Veritabanı Dosyasının Yolu:** CSV dosyanızın tam yolunu girin (örneğin, `/path/to/your/data.csv` veya `C:\\Users\\YourUser\\Documents\\data.csv`).
    *   **Bağımlı Değişkenin Adı:** Tahmin etmek istediğiniz hedef sütunun adını girin.

4.  **Google Drive Bağlantısı (Google Colab için):**
    Eğer kodu Google Colab üzerinde çalıştırıyorsanız, `drive.mount('/content/drive')` satırı Google Drive'ınızı Colab ortamına bağlamak için kullanılır. Veri setiniz Google Drive'da ise bu adımı uygulamanız gerekecektir. Yerel bir ortamda çalışıyorsanız bu adıma gerek yoktur ve ilgili satır kaldırılabilir.

## Çıktılar

Betiğin çalışması tamamlandığında aşağıdaki çıktılar üretilecektir:

*   **Veri Bilgisi ve Tanımlayıcı İstatistikler:** Yüklenen veri setinin genel bilgileri ve sayısal sütunlar için tanımlayıcı istatistikler konsola yazdırılır.
*   **Model Performans Sonuçları:**
    *   Karar Ağacı Regresyonu, Rastgele Orman Regresyonu ve Gradyan Artırma Regresyonu modellerinin R², MSE ve RMSE değerlerini içeren bir tablo konsola yazdırılır.
*   **Görselleştirmeler:**
    *   **Modellerin Performansı:** Farklı modellerin R² skorlarını karşılaştıran bir çubuk grafik.
    *   **Gerçek vs Tahmin Edilen Değerler:** Test seti için gerçek değerlere karşılık modelin tahmin ettiği değerleri gösteren bir dağılım grafiği.
    *   **Model Hata Dağılımı:** Modelin tahmin hatalarının (artıklar) dağılımını gösteren bir histogram.
    *   **Confusion Matrix:** Regresyon sonuçlarının belirli eşiklere göre kategorize edilmesiyle oluşturulan bir karmaşıklık matrisi. Bu, modelin farklı değer aralıklarındaki performansını anlamaya yardımcı olabilir.

## Örnek Kod Akışı

Mevcut `README.md` dosyasındaki kod, bu projenin temel mantığını içermektedir. Aşağıda bu kodun ana adımları özetlenmiştir:

1.  Gerekli kütüphaneler içe aktarılır.
2.  Kullanıcıdan veri seti yolu ve hedef değişken adı alınır.
3.  Veri seti yüklenir ve temel bilgiler gösterilir.
4.  Sayısal ve kategorik sütunlar ayrılır.
5.  Eksik değerler uygun stratejilerle (sayısal için ortalama, kategorik için en sık tekrar eden) doldurulur.
6.  Kategorik değişkenler one-hot encoding ile sayısal forma dönüştürülür.
7.  Bağımsız (X) ve bağımlı (y) değişkenler ayrılır.
8.  Veri, eğitim ve test kümelerine bölünür.
9.  (İsteğe bağlı olarak) Veri ölçeklendirilir (Karar ağaçları için genellikle gerekli değildir ancak kodda bulunmaktadır).
10. Karar Ağacı, Rastgele Orman ve Gradyan Artırma regresyon modelleri tanımlanır.
11. Her bir model eğitilir, test seti üzerinde tahminler yapılır ve performans metrikleri (R², MSE, RMSE) hesaplanır.
12. Hesaplanan metrikler bir tablo halinde sunulur.
13. Performans metrikleri ve tahmin sonuçları çeşitli grafiklerle görselleştirilir.

## Katkıda Bulunma

Katkılarınız her zaman beklerim! Lütfen bir "issue" açarak veya bir "pull request" göndererek katkıda bulunun.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına bakınız (Eğer varsa).
