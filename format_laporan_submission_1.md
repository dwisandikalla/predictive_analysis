# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek

Penyakit diabetes merupakan salah satu masalah kesehatan global yang semakin meningkat prevalensinya setiap tahun. Menurut laporan dari World Health Organization (WHO), lebih dari 422 juta orang di seluruh dunia menderita diabetes, dan angka ini diperkirakan akan terus meningkat secara signifikan dalam beberapa dekade ke depan [1](https://www.who.int/news-room/fact-sheets/detail/diabetes). Diabetes, khususnya tipe 2, dapat menyebabkan komplikasi serius seperti penyakit jantung, gagal ginjal, kebutaan, hingga kematian dini jika tidak terdeteksi dan ditangani sejak dini.

Di Indonesia sendiri, berdasarkan data dari Riskesdas (Riset Kesehatan Dasar) 2018, prevalensi diabetes melitus yang terdiagnosis oleh tenaga kesehatan maupun tidak terdiagnosis terus meningkat, yakni dari 6,9% pada tahun 2013 menjadi 10,9% pada tahun 2018 [2](https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas/). Hal ini menandakan bahwa banyak kasus diabetes tidak terdeteksi sejak awal, yang pada akhirnya menimbulkan beban ekonomi dan kesehatan yang lebih besar.

Dalam konteks ini, penerapan analisis prediktif berbasis data menjadi semakin penting. Dengan memanfaatkan teknologi machine learning dan data kesehatan seperti tekanan darah, kadar glukosa, insulin, hingga indeks massa tubuh (BMI), kita dapat membangun sistem prediksi yang mampu mengidentifikasi potensi risiko diabetes pada seseorang dengan akurasi yang tinggi.

Proyek ini menggunakan dataset Healthcare-Diabetes.csv yang berisi 2.768 observasi dan 10 variabel, termasuk variabel target Outcome yang merepresentasikan diagnosis diabetes (1 = diabetes, 0 = tidak diabetes). Model prediktif dikembangkan menggunakan tiga algoritma machine learning, yaitu Linear Regression, Random Forest, dan K-Nearest Neighbors (KNN). Evaluasi dilakukan menggunakan metrik Mean Squared Error (MSE) untuk menilai kinerja masing-masing model dalam memprediksi risiko diabetes.

Melalui pendekatan ini, diharapkan tercipta sistem deteksi dini diabetes yang akurat dan efisien, yang dapat membantu lembaga kesehatan, klinik, maupun individu dalam mengambil keputusan preventif sebelum kondisi diabetes berkembang menjadi kronis.

## Business Understanding

Penerapan machine learning dalam bidang kesehatan membuka peluang besar dalam upaya preventif terhadap penyakit kronis seperti diabetes. Dalam konteks ini, pemahaman yang kuat terhadap tujuan bisnis dan formulasi masalah sangat penting untuk mengarahkan proses analisis data dan pembangunan model prediktif yang akurat.

### Problem Statements

- Bagaimana mengidentifikasi pasien yang berisiko tinggi terkena diabetes berdasarkan data kesehatan dasar seperti kadar glukosa, tekanan darah, BMI, dan usia?
- Model machine learning apa yang paling optimal dalam memprediksi risiko diabetes berdasarkan performa metrik Mean Squared Error (MSE)?
- Apakah preprocessing data seperti standarisasi dan penanganan outlier dapat meningkatkan performa model prediktif secara signifikan?

### Goals

- Mengembangkan sistem prediktif yang mampu mengidentifikasi pasien berisiko diabetes hanya dengan data medis dasar.
- Mengevaluasi dan membandingkan performa beberapa model prediktif, yaitu Linear Regression, Random Forest, dan K-Nearest Neighbors, berdasarkan metrik MSE.
- Menilai dampak preprocessing data terhadap performa model. 

### Solution statements

- Menggunakan lebih dari satu algoritma machine learning (Linear Regression, Random Forest, KNN) untuk membangun model prediktif.
- Melakukan preprocessing data melalui tahapan:
  - Penghapusan outlier dengan metode Interquartile Range (IQR)
  - Standarisasi fitur numerik menggunakan StandardScaler
  - Penghapusan kolom yang tidak relevan (misal kolom Id)

## Data Understanding
Dataset yang digunakan dalam proyek ini merupakan dataset Healthcare-Diabetes yang bersumber dari platform Kaggle. Dataset ini berisi data medis dasar yang sering dijadikan indikator risiko diabetes pada pasien. Dataset dapat diunduh melalui tautan berikut: [Diabetes Dataset]([https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes)). Dataset ini terdiri dari 2.768 baris dan 10 kolom, dengan 9 kolom sebagai fitur dan 1 kolom target (label). Data ini digunakan untuk memprediksi kemungkinan seseorang menderita diabetes berdasarkan beberapa fitur kesehatan.

### Variabel-variabel pada Diabetes dataset adalah sebagai berikut:
- Id: Nomor unik identifikasi pasien. Kolom ini dihapus karena tidak memiliki nilai prediktif.
- Pregnancies: Jumlah kehamilan yang pernah dialami oleh pasien.
- Glucose: Kadar glukosa dalam darah.
- BloodPressure: Tekanan darah diastolik (mm Hg).
- SkinThickness: Ketebalan lipatan kulit trisep (mm).
- Insulin: Kadar insulin dalam darah (mu U/ml).
- BMI: Indeks Massa Tubuh, dihitung sebagai berat (kg) dibagi kuadrat tinggi badan (m²).
- DiabetesPedigreeFunction: Skor riwayat keluarga penderita diabetes (faktor genetik).
- Age: Usia pasien (dalam tahun).
- Outcome: Variabel target, bernilai 1 jika pasien menderita diabetes dan 0 jika tidak.

### Exploratory Data Analysis (EDA)


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

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

