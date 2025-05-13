# Laporan Proyek Machine Learning - Dwi Sandi Kalla

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang menjadi masalah kesehatan global. Diabetes terjadi ketika pankreas tidak memproduksi insulin dalam jumlah cukup atau ketika tubuh tidak dapat menggunakan insulin yang diproduksi dengan baik. Menurut laporan dari World Health Organization (WHO), pada tahun 2021 diabetes merupakan penyebab langsung dari 1,6 juta kematian dan 47% dari semua kematian akibat diabetes dialami oleh orang sebelum berusia 70 tahun.[[1]](https://www.who.int/news-room/fact-sheets/detail/diabetes). Seiring waktu, diabetes dapat merusak sistem pembuluh darah di jantung, mata, ginjal, dan saraf. Diabetes dapat menyebabkan hilangnya penglihatan secara permanen karena rusaknya pembuluh darah di mata. Banyak penderita diabetes mengalami masalah pada kaki karena kerusakan saraf dan aliran darah yang cenderung buruk. Hal ini menyebabkan terjadinya borok kaki dan dapat berujung pada amputasi.

Deteksi dini terhadap risiko diabetes dapat menjadi langkah yang sangat awal untuk mengurangi penyakit lain yang disebabkan oleh diabetes. Dalam hal ini, pendekatan berbasis data dan Artificial Intelligence (AI) menjadi salah satu solusi yang dapat diandalkan. Perkembangan teknologi machine learning dan data science telah membuka peluang baru untuk menganalisis data kesehatan dalam skala besar dan menghasilkan model prediksi yang akurat.

Proyek ini bertujuan untuk membangun model prediksi diabetes dengan memanfaatkan Diabetes Prediction Dataset yang saya peroleh dari kaggle dan berisi berbagai indikator kesehatan seperti jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, kadar insulin, indeks massa tubuh (Body Mass Index), umur, serta faktor genetik melalui Diabetes Pedigree Function. Dataset ini disusun untuk mendukung pengembangan model prediktif yang mampu mengidentifikasi individu dengan risiko tinggi terkena diabetes secara lebih akurat dan efisien.

Dengan menerapkan algoritma machine learning seperti K-Nearest Neighbors, Random Forest, Linear Regression, proyek ini akan mengevaluasi performa masing-masing model berdasarkan metrik Mean Squared Error (MSE). Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang efektif, namun juga dapat memberikan wawasan bagi praktisi kesehatan dalam pengambilan keputusan yang berbasis data.

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

- Menerapkan beberapa algoritma machine learning seperti K-Nearest Neighbors (KNN), Random Forest, dan Linear Regression untuk membangun model prediksi diabetes, dengan melakukan pembandingan performa menggunakan metrik evaluasi Mean Squared Error (MSE) pada data latih dan uji
- Melakukan pembersihan data secara menyeluruh sebelum pelatihan model, mencakup penanganan missing values, penghapusan duplikat, dan deteksi outlier menggunakan teknik seperti IQR (Interquartile Range).

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah [Healthcare Diabetes Dataset](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes) yang berasal dari kaggle yang dirancang untuk keperluan prediksi risiko diabetes berdasarkan beberapa faktor. Dataset ini terdiri dari 2768 baris dan 9 kolom, dimana setiap baris merepresentasikan satu individu. Data ini memiliki satu variabel target yang bersifat biner, yaitu "Outcome" yang menunjukkan seseorang mengidap diabetes (1) atau tidak (0).

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- Id : 	Pengidentifikasi unik untuk setiap baris data atau individu. Kolom ini bersifat administratif dan tidak memiliki peran langsung dalam proses prediksi. Biasanya akan dihapus pada tahap preprocessing.
- _Pregnancies_ (Kehamilan) : Menunjukkan jumlah kehamilan yang pernah dialami oleh pasien wanita.
- _Glucose_ (Glukosa) : Menggambarkan kadar glukosa plasma dalam darah setelah 2 jam dalam uji toleransi glukosa oral.
- _BloodPressure_ (Tekanan Darah) : Merujuk pada tekanan darah diastolik, yaitu tekanan saat jantung berada dalam keadaan rileks di antara dua detakan. Tekanan darah tinggi sering ditemukan bersamaan dengan diabetes dan termasuk dalam sindrom metabolik.
- _SkinThickness_ (Ketebalan Kulit) : Mengukur ketebalan lipatan kulit triceps, yang dapat digunakan sebagai indikator lemak tubuh.
- Insulin : Menunjukkan kadar insulin serum 2 jam setelah tes toleransi glukosa. Kadar insulin dapat membantu menilai resistensi insulin yang merupakan akar penyebab utama diabetes.
- BMI : Singkatan dari _Body Mass Index_, yaitu rasio berat badan terhadap tinggi badan dalam meter kuadrat. BMI merupakan indikator yang umum digunakan untuk mengklasifikasikan individu sebagai kurus, normal, gemuk, atau obesitas — faktor penting dalam prediksi diabetes.
- DiabetesPedigreeFunction : Skor yang mengindikasikan faktor keturunan/genetik risiko diabetes. Nilai ini dihitung berdasarkan riwayat diabetes dalam keluarga dan kompleksitas hubungan kekerabatan.
- _Age_ (Usia) : Usia pasien dalam tahun.
- _Outcome_ (Hasil) : **1** Individu tersebut menderita diabetes, **2** Individu tersebut tidak menderita diabetes.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

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

