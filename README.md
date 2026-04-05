# 📈 Peramalan Harga Saham TEBE (ARIMA Model)

Proyek ini merupakan analisis deret waktu (*time series*) untuk memprediksi pergerakan harga saham PT Dana Brata Luhur Tbk (TEBE) dalam jangka pendek (40 hari ke depan). Pemodelan dilakukan menggunakan pendekatan metodologi Box-Jenkins dengan model **ARIMA/ARMA**.

## 🎯 Tujuan Proyek
* Menganalisis pola historis harga penutupan saham TEBE selama 5 tahun terakhir.
* Mengidentifikasi dan mengestimasi model ARIMA terbaik berdasarkan kriteria *Akaike Information Criterion* (AIC).
* Memvalidasi kelayakan model melalui uji diagnostik *Ljung-Box*.
* Menyediakan proyeksi pergerakan harga saham beserta interval kepercayaan 95%.

## 📂 Struktur Repositori
Repositori ini disusun agar mudah ditelusuri:

* 📁 **`/analysis`** — Berisi *source code* R Markdown (`.Rmd`) dan **[Laporan Teknis Lengkap beserta Visualisasinya](analysis/arima_tebe.md)**.
* 📁 **`/output`** — Berisi artefak akhir proyek (Model `.rds` yang siap di-*deploy*, tabel data *forecast* format CSV, dan plot resolusi tinggi).
* 📁 **`/renv`** — *Environment manager* untuk memastikan reproduksibilitas *library* R yang digunakan.

## 📊 Hasil Utama
Model terbaik yang terpilih adalah **ARIMA(2,1,1)**. Model ini berhasil melewati uji diagnostik (residual bersifat *white noise*) dan menghasilkan *Mean Absolute Percentage Error* (MAPE) sebesar 32.77% pada data *testing*. 

👉 **[Klik di sini untuk membaca laporan analisis dan kodingan selengkapnya.](analysis/arima_tebe.md)**

---
*Dibuat menggunakan R dan divisualisasikan dengan paket `forecast`.*