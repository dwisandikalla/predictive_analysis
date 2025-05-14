# Laporan Proyek Machine Learning - Dwi Sandi Kalla

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang menjadi masalah kesehatan global. Diabetes terjadi ketika pankreas tidak memproduksi insulin dalam jumlah cukup atau ketika tubuh tidak dapat menggunakan insulin yang diproduksi dengan baik. Menurut laporan dari _World Health Organization_ (WHO), pada tahun 2021 diabetes merupakan penyebab langsung dari 1,6 juta kematian dan 47% dari semua kematian akibat diabetes dialami oleh orang sebelum berusia 70 tahun.[[1]](https://www.who.int/news-room/fact-sheets/detail/diabetes). Seiring waktu, diabetes dapat merusak sistem pembuluh darah di jantung, mata, ginjal, dan saraf. Diabetes dapat menyebabkan hilangnya penglihatan secara permanen karena rusaknya pembuluh darah di mata. Banyak penderita diabetes mengalami masalah pada kaki karena kerusakan saraf dan aliran darah yang cenderung buruk. Hal ini menyebabkan terjadinya borok kaki dan dapat berujung pada amputasi.

Deteksi dini terhadap risiko diabetes dapat menjadi langkah yang sangat awal untuk mengurangi penyakit lain yang disebabkan oleh diabetes. Dalam hal ini, pendekatan berbasis data dan _Artificial Intelligence_ (AI) menjadi salah satu solusi yang dapat diandalkan. Perkembangan teknologi machine learning dan data science telah membuka peluang baru untuk menganalisis data kesehatan dalam skala besar dan menghasilkan model prediksi yang akurat.

Proyek ini bertujuan untuk membangun model prediksi diabetes dengan memanfaatkan Diabetes Prediction Dataset yang saya peroleh dari kaggle dan berisi berbagai indikator kesehatan seperti jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, indeks massa tubuh (_Body Mass Index_), umur, serta faktor genetik melalui _Diabetes Pedigree Function_. Dataset ini disusun untuk mendukung pengembangan model prediktif yang mampu mengidentifikasi individu dengan risiko tinggi terkena diabetes secara lebih akurat dan efisien.

Dengan menerapkan algoritma machine learning seperti K-Nearest Neighbors, Random Forest, Linear Regression, proyek ini akan mengevaluasi performa masing-masing model berdasarkan metrik _Mean Squared Error_ (MSE). Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang efektif, namun juga dapat memberikan wawasan bagi praktisi kesehatan dalam pengambilan keputusan yang berbasis data.

Menurut studi oleh Pérez-Gandía et al. (2018), penerapan sistem berbasis AI dalam prediksi diabetes mampu meningkatkan akurasi diagnosis hingga 87% dibanding metode konvensional [[2]](https://www.jmir.org/2018/5/e10775/). Hal ini menunjukkan potensi besar dari pendekatan teknologi dalam meningkatkan layanan kesehatan preventif. Dengan latar belakang tersebut, pengembangan model prediksi diabetes berbasis machine learning menjadi suatu kebutuhan penting dalam upaya deteksi dini dan pengendalian penyakit diabetes, khususnya dalam masyarakat yang belum memiliki akses terhadap pemeriksaan kesehatan rutin.

## Business Understanding

### Problem Statements

- Bagaimana proses pembersihan data seperti penanganan missing value, outlier, dan duplikat dapat meningkatkan kualitas dataset serta membantu menghasilkan model prediksi diabetes yang lebih akurat dan andal?
- Bagaimana membangun model prediksi untuk mengidentifikasi risiko seseorang terkena diabetes berdasarkan data jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, indeks massa tubuh (Body Mass Index), umur, serta faktor genetik melalui Diabetes Pedigree Function?
- Algoritma machine learning mana yang memberikan performa terbaik berdasarkan nilai MSE dalam memprediksi diabetes pada dataset yang digunakan?

### Goals

- Melakukan proses pembersihan data melalui identifikasi dan penanganan missing values, outlier, serta data duplikat guna meningkatkan kualitas data input, sehingga model machine learning dapat belajar dari informasi yang bersih dan representatif untuk memprediksi risiko diabetes.
- Mengembangkan model prediktif berbasis data kesehatan untuk mengidentifikasi individu dengan risiko diabetes secara akurat.
- Membandingkan performa beberapa algoritma machine learning seperti K-Nearest Neighbors, Random Forest, dan Linear Regression menggunakan metrik Mean Squared Error (MSE).

### Solution Statement

- Menerapkan beberapa algoritma machine learning seperti K-Nearest Neighbors (KNN), Random Forest, dan Linear Regression untuk membangun model prediksi diabetes, dengan melakukan pembandingan performa menggunakan metrik evaluasi _Mean Squared Error_ (MSE) pada data latih dan uji
- Melakukan pembersihan data secara menyeluruh sebelum pelatihan model, mencakup penanganan missing values, penghapusan duplikat, dan deteksi outlier menggunakan teknik seperti IQR (Interquartile Range).

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [_Healthcare Diabetes Dataset_](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes) yang berasal dari kaggle yang dirancang untuk keperluan prediksi risiko diabetes berdasarkan beberapa faktor. Dataset ini terdiri dari 2768 baris dan 9 kolom, dimana setiap baris merepresentasikan satu individu. Data ini memiliki satu variabel target yang bersifat biner, yaitu "Outcome" yang menunjukkan seseorang mengidap diabetes (1) atau tidak (0).

### Variabel-variabel pada _Healthcare Diabetes Dataset_ adalah sebagai berikut:
- Id : 	Pengidentifikasi unik untuk setiap baris data atau individu. Kolom ini bersifat administratif dan tidak memiliki peran langsung dalam proses prediksi. Biasanya akan dihapus pada tahap preprocessing.
- _Pregnancies_ (Kehamilan) : Menunjukkan jumlah kehamilan yang pernah dialami oleh pasien wanita.
- _Glucose_ (Glukosa) : Menggambarkan kadar glukosa plasma dalam darah setelah 2 jam dalam uji toleransi glukosa oral.
- _BloodPressure_ (Tekanan Darah) : Merujuk pada tekanan darah diastolik, yaitu tekanan saat jantung berada dalam keadaan rileks di antara dua detakan. Tekanan darah tinggi sering ditemukan bersamaan dengan diabetes dan termasuk dalam sindrom metabolik.
- _SkinThickness_ (Ketebalan Kulit) : Mengukur ketebalan lipatan kulit triceps, yang dapat digunakan sebagai indikator lemak tubuh.
- Insulin : Menunjukkan kadar insulin serum 2 jam setelah tes toleransi glukosa. Kadar insulin dapat membantu menilai resistensi insulin yang merupakan akar penyebab utama diabetes.
- BMI : Singkatan dari _Body Mass Index_, yaitu rasio berat badan terhadap tinggi badan dalam meter kuadrat. BMI merupakan indikator yang umum digunakan untuk mengklasifikasikan individu sebagai kurus, normal, gemuk, atau obesitas — faktor penting dalam prediksi diabetes.
- _DiabetesPedigreeFunction_ : Skor yang mengindikasikan faktor keturunan/genetik risiko diabetes. Nilai ini dihitung berdasarkan riwayat diabetes dalam keluarga dan kompleksitas hubungan kekerabatan.
- _Age_ (Usia) : Usia pasien dalam tahun.
- _Outcome_ (Hasil) : **1** Individu tersebut menderita diabetes, **2** Individu tersebut tidak menderita diabetes.

### Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) adalah proses memahami struktur, pola, dan anomali dari sebuah dataset sebelum dilakukan pemodelan. Pada proyek ini dilakukan beberapa tahapan EDA sebagai berikut:
1. Pemeriksaan Awal pada Data
   - Memeriksa fitur data dengan `data.info()`
     <br>Digunakan untuk melihat struktur dataset, termasuk jumlah baris dan kolom, tipe data, dan apakah ada nilai yang hilang. Hasilnya menunjukkan bahwa semua kolom memiliki tipe data numerik dan tidak terdapat missing values (kosong) dengan total kolom 9 dan baris 2768.
   - Mengecek duplikasi data `data.duplicated().sum()`
     <br>Mengecek apakah ada data yang duplikat atau tidak. Untuk hasil pada pengecekan duplikasi, data tidak memiliki nilai duplikat.
   - Memastikan tidak ada missing value `data.isna().sum()`
     <br>Memastikan bahwa tidak ada nilai _NaN_ untuk setiap kolom yang menandakan missing value.
   - Penghapusan kolom **Id**
     <br>Kolom **Id** dihapus karena tidak relevan terhadap proses prediksi, hanya berfungsi sebagai identifikasi unik.
2. Statistika Deskriptif
3. Deteksi Outlier dengan Boxplot
5. Penghapusan Outlier dengan IQR Method
6. Distribusi Fitur (Histogram)
7. Pairplot (Visualisasi Hubungan antar Variabel)
8. Heatmap Korelasi

## Data Preparation
Tahap Data Preparation merupakan langkah penting sebelum melakukan proses training model machine learning. Tujuannya adalah untuk memastikan data dalam kondisi optimal agar model dapat belajar secara efektif. Berikut ini adalah tahapan data preparation yang dilakukan dalam proyek ini:
1. Mengecek Ringkasan Informasi Dataset
   - Mengecek informasi data menggunakan `data.info()`.
   - Tujuan dari pengecekan ini adalah untuk membantu memahami struktur data, Mengidentifikasi nilai yang hilang, memeriksa setiap type data dari setiap kolom, dan juga termasuk langkah awal untuk proses _data cleaning_.
2. Mengecek Duplikasi Data
   - Mengecek duplikasi data dilakukan dengan kode `data.duplicated().sum()`.
   - Pada proyek ini tidak ditemukan adanya data yang duplikat.
   - Mengecek duplikasi data bertujuan agar data tidak ganda, data ganda dapat mendominasi hasil perhitungan statistik yang menghasilkan kesimpulan yang bias. Proses pengecekan duplikasi diperlukan untuk mendapatkan representasi data yang akurat, efisien, dan relevan untuk pengambilan keputusan yang tepat.
3. Pemeriksaan dan Penanganan Nilai Kosong (Missing Values)
   - Mengecek missing value dapat menggunakan `data.isna().sum()`
   - Pada proyek ini tidak ditemukan adanya missing value.
   - Nilai kosong dapat mengganggu proses pelatihan model. Jika ada, harus dilakukan penanganan seperti imputasi (pengisian nilai) atau penghapusan baris/kolom.
4. Penghapusan Kolom **Id**
   - Penghapusan kolom **Id** dilakukan melalui kode berikut `data.drop('Id', axis=1, inplace=True)`.
   - Kolom **Id** hanya berisi identitas pasien dan tidak memiliki kontribusi terhadap prediksi diabetes. Kolom seperti ini disebut irrelevant feature dan dapat menyebabkan noise dalam proses pelatihan model.
5. Pengecekan Outlier dengan Box-plot
   - Pengecekan outlier menggunakan boxplot untuk setiap kolom dilakukan dengan menggunakan
     ```python
     for column in data.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f'Boxplot for {column}')
        plt.show()
     ```
   - Hasilnya setiap kolom memiliki nilai outlier kecuali kolom Outcome.
   -  Beberapa kolom/fitur memiliki nilai outlier yang jika tidak ditangani, outlier bisa menyebabkan model belajar pola yang tidak benar (overfitting atau bias).
   -  > Gambar dapat dilihat di : [Google Collabs - project](https://colab.research.google.com/drive/1XQ_spIaupa-1KupVIS4ozF7IhCmrcStA?usp=sharing)
6. Menangani Outlier (IQR)
   - Penanganan dilakukan dengan metode Interquartile Range (IQR) yang dilakukan memalui kode berikut:
     ```python
     Q1 = data.quantile(0.25)
     Q3 = data.quantile(0.75)
     IQR = Q3 - Q1

     # Menghapus baris yang mengandung outlier
     data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
     data.info()
     ```
   - Setelah penghapusan outlier jumlah baris yang semula 2768 menjadi 2299 baris.
   - Alasan dilakukan penerapan metode IQR adalah karena ingin menghapus outlier agar nantinya tidak berpengaruh ke model.
7. Distribusi Fitur (Histogram)
   - Disitribusi fitur dilakukan melalui pembuatan histogram dengan kode berikut :
     ```python
     data.hist(bins=50, figsize=(20,15))
     plt.show()
     ```
   - Hasil dari distribusi nya adalah sebagai berikut:
     * Pregnancies, Age, DiabetesPedigreeFunction, SkinThickness, Insulin: Distribusinya miring ke kanan (right-skewed).
     * Glucose & BMI: Hampir normal, sedikit miring ke kanan.
     * BloodPressure: Simetris, mendekati normal.
     * Outcome: Data biner dan tidak seimbang (lebih banyak kelas 0).
    - Alasan dilakukannya ini adalah untuk mencari temuan baru terkait data.
8. Korelasi Antar Fitur
   - Menggunakan correlation matrix dan pairplot
   - Hasil dari correlation matrix adalah
     * Glucose punya korelasi paling tinggi dengan Outcome (0.50) → fitur penting untuk prediksi diabetes.
     * Age dan Pregnancies berkorelasi tinggi (0.58) → makin tua, makin banyak kehamilan.
     * SkinThickness dan Insulin berkorelasi cukup kuat (0.49) → sama-sama terkait metabolisme.
     * Fitur BMI, Age, dan Pregnancies punya korelasi sedang ke Outcome (sekitar 0.2–0.3).
     * Korelasi digunakan untuk melihat hubungan antar fitur dan memilih fitur yang relevan untuk model.
    - Analisis korelasi digunakan untuk menentukan fitur yang relevan dalam pemodelan, di mana fitur dengan korelasi tinggi terhadap variabel target (seperti Outcome) layak dipertahankan karena berkontribusi signifikan. Selain itu, korelasi juga membantu menghindari multikolinearitas, yaitu kondisi ketika dua fitur memiliki hubungan yang sangat kuat (misalnya antara SkinThickness dan Insulin), yang dapat mengganggu stabilitas model. Meskipun begitu, fitur dengan korelasi rendah bukan berarti tidak penting, karena kombinasi antar fitur tetap dapat meningkatkan performa model secara keseluruhan.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

