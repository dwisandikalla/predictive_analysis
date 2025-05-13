# Laporan Proyek Machine Learning - Dwi Sandi Kalla

## Domain Proyek

Diabetes melitus merupakan salah satu penyakit tidak menular yang menjadi masalah kesehatan global. Menurut laporan dari World Health Organization (WHO), pada tahun 2021 diperkirakan lebih dari 422 juta orang di seluruh dunia hidup dengan diabetes, dan angka ini terus meningkat setiap tahunnya [1](https://www.who.int/news-room/fact-sheets/detail/diabetes). Diabetes tipe 2, yang merupakan jenis diabetes paling umum, sering kali tidak terdiagnosis pada tahap awal karena gejalanya yang tidak spesifik. Padahal, diagnosis dini sangat penting agar pasien dapat segera mendapatkan penanganan medis dan melakukan perubahan gaya hidup yang diperlukan.

Deteksi dini terhadap risiko diabetes dapat menjadi langkah preventif untuk mengurangi komplikasi jangka panjang seperti penyakit jantung, kerusakan ginjal, kebutaan, hingga amputasi. Dalam hal ini, pendekatan berbasis data menjadi salah satu solusi yang dapat diandalkan. Perkembangan teknologi machine learning dan data science telah membuka peluang baru untuk menganalisis data kesehatan dalam skala besar dan menghasilkan model prediksi yang akurat.

Proyek ini bertujuan untuk membangun model prediksi diabetes dengan memanfaatkan Diabetes Prediction Dataset, yang berisi berbagai indikator kesehatan seperti jumlah kehamilan, kadar glukosa, tekanan darah, indeks massa tubuh (Body Mass Index), serta faktor genetik melalui Diabetes Pedigree Function. Dataset ini disusun untuk mendukung pengembangan model prediktif yang mampu mengidentifikasi individu dengan risiko tinggi terkena diabetes secara lebih akurat dan efisien.

Dengan menerapkan algoritma machine learning seperti K-Nearest Neighbors, Random Forest, Linear Regression, dan AdaBoost, proyek ini akan mengevaluasi performa masing-masing model berdasarkan metrik Mean Squared Error (MSE) dan melakukan optimasi melalui hyperparameter tuning. Proyek ini diharapkan tidak hanya menghasilkan model prediktif yang efektif, namun juga dapat memberikan wawasan bagi praktisi kesehatan dalam pengambilan keputusan yang berbasis data.

Menurut studi oleh Pérez-Gandía et al. (2018), penerapan sistem berbasis AI dalam prediksi diabetes mampu meningkatkan akurasi diagnosis hingga 87% dibanding metode konvensional [2]. Hal ini menunjukkan potensi besar dari pendekatan teknologi dalam meningkatkan layanan kesehatan preventif.

Dengan latar belakang tersebut, pengembangan model prediksi diabetes berbasis machine learning menjadi suatu kebutuhan penting dalam upaya deteksi dini dan pengendalian penyakit diabetes, khususnya dalam masyarakat yang belum memiliki akses terhadap pemeriksaan kesehatan rutin.

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1
- Pernyataan Masalah 2
- Pernyataan Masalah n

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban pernyataan masalah 1
- Jawaban pernyataan masalah 2
- Jawaban pernyataan masalah n

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut: 

    ### Solution statements
    - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
    - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

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

