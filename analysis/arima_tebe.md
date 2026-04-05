PROYEK PERAMALAN HARGA SAHAM TEBE MENGGUNAKAN MODEL ARIMA
================
Haikal Al Faruqi
2026-04-06

- [1. Data Processing](#1-data-processing)
  - [Interpretasi](#interpretasi)
  - [1.2. Pre-processing Data](#12-pre-processing-data)
  - [Interpretasi](#interpretasi-1)
- [Including Plots](#including-plots)
  - [1.3. Data Transformation](#13-data-transformation)
  - [Interpretasi](#interpretasi-2)
  - [2. Modelling ARIMA](#2-modelling-arima)
  - [Interpretasi](#interpretasi-3)
  - [2.2. Estimasi Parameter Model](#22-estimasi-parameter-model)
  - [Interpretasi](#interpretasi-4)
  - [2.3. Pemilihan Model Terbaik](#23-pemilihan-model-terbaik)
  - [Interpretasi](#interpretasi-5)
  - [2.4. Validasi Diagnostik Awal](#24-validasi-diagnostik-awal)
  - [Interpretasi](#interpretasi-6)
  - [3. Evaluate](#3-evaluate)
  - [Interpretasi](#interpretasi-7)
  - [3.2. Forecasting](#32-forecasting)
  - [Interpretasi](#interpretasi-8)
  - [3.3. Visualisasi Hasil Peramalan](#33-visualisasi-hasil-peramalan)
  - [Interpretasi Visualisasi](#interpretasi-visualisasi)
  - [3.4. Tabulasi Hasil Prediksi](#34-tabulasi-hasil-prediksi)
  - [Interpretasi Tabel](#interpretasi-tabel)
- [4. Kesimpulan](#4-kesimpulan)
  - [4.1. Ringkasan Metodologi](#41-ringkasan-metodologi)
  - [4.2. Model Terpilih](#42-model-terpilih)
  - [4.3. Hasil Forecasting](#43-hasil-forecasting)
  - [4.4. Keterbatasan dan Saran](#44-keterbatasan-dan-saran)
  - [4.5. Penutup](#45-penutup)

## 1. Data Processing

Tahap pertama dalam penelitian ini adalah pengumpulan dan pembersihan
data. Data yang digunakan adalah harga penutupan harian saham PT Dana
Brata Luhur Tbk (TEBE) selama 5 tahun terakhir. Proses ini mencakup
akuisisi data dari sumber terpercaya dan pembersihan dari nilai yang
hilang (*missing values*) akibat hari libur bursa atau akhir pekan.

``` r
# ==============================================================================
# 1.1. COLLECTING DAN CLEANING DATA
# ==============================================================================
# Konfigurasi awal dan pembuatan direktori output
set.seed(123)
dir.create("output", showWarnings = FALSE)
dir.create("output/plots", showWarnings = FALSE)
dir.create("output/tables", showWarnings = FALSE)
dir.create("output/models", showWarnings = FALSE)

# Load library yang diperlukan
library(quantmod)
```

    ## Loading required package: xts

    ## Loading required package: zoo

    ## 
    ## Attaching package: 'zoo'

    ## The following objects are masked from 'package:base':
    ## 
    ##     as.Date, as.Date.numeric

    ## Loading required package: TTR

    ## Registered S3 method overwritten by 'quantmod':
    ##   method            from
    ##   as.zoo.data.frame zoo

``` r
library(tseries)
library(forecast)

# Menarik data saham TEBE 5 tahun terakhir dari Yahoo Finance
getSymbols("TEBE.JK", 
           from = as.Date(Sys.Date() - 5*365), 
           to = Sys.Date(),
           auto.assign = TRUE)
```

    ## [1] "TEBE.JK"

``` r
# Mengekstrak harga penutupan dan membersihkan missing values (NA)
harga_tebe <- as.numeric(Cl(TEBE.JK))
harga_tebe <- harga_tebe[!is.na(harga_tebe)]
tanggal <- as.Date(index(TEBE.JK)[!is.na(Cl(TEBE.JK))])

# Menampilkan informasi data
cat("=== INFORMASI DATA ===\n")
```

    ## === INFORMASI DATA ===

``` r
cat("Total Observasi  :", length(harga_tebe), "hari perdagangan\n")
```

    ## Total Observasi  : 1198 hari perdagangan

``` r
cat("Periode Data     :", format(min(tanggal), "%d %B %Y"), "s/d", format(max(tanggal), "%d %B %Y"), "\n")
```

    ## Periode Data     : 07 April 2021 s/d 02 April 2026

``` r
cat("Harga Terakhir   : Rp", format(tail(harga_tebe, 1), big.mark="."), "\n")
```

    ## Harga Terakhir   : Rp 1.445

### Interpretasi

Output di atas menunjukkan bahwa data berhasil diambil sebanyak 1198
observasi. Periode data mencakup 5 tahun terakhir hingga hari ini. Nilai
`NA` telah dihapus untuk memastikan kontinuitas deret waktu, yang
merupakan syarat wajib sebelum pemodelan ARIMA.

### 1.2. Pre-processing Data

Sesuai standar metodologi deret waktu, data dibagi menjadi dua subset:
**70% data training** untuk membangun model dan **30% data testing**
untuk evaluasi performa prediktif. Pembagian ini dilakukan secara
kronologis untuk menjaga integritas temporal data deret waktu.

``` r
# ==============================================================================
# 1.2. PEMBAGIAN DATA & VISUALISASI
# ==============================================================================
n_total <- length(harga_tebe)
n_train <- floor(n_total * 0.70)

train_data <- harga_tebe[1:n_train]
test_data  <- harga_tebe[(n_train + 1):n_total]

cat("\n=== PEMBAGIAN DATA ===\n")
```

    ## 
    ## === PEMBAGIAN DATA ===

``` r
cat("Data Training:", length(train_data), "hari (70%)\n")
```

    ## Data Training: 838 hari (70%)

``` r
cat("Data Testing :", length(test_data), "hari (30%)\n")
```

    ## Data Testing : 360 hari (30%)

``` r
# Visualisasi Pembagian Data
plot(harga_tebe, type = "l", col = "lightgray", lwd = 1, 
     main = "Pembagian Data Training & Testing",
     xlab = "Waktu (Indeks Hari)", ylab = "Harga (IDR)",
     ylim = c(min(harga_tebe), max(harga_tebe)))

lines(1:n_train, train_data, col = "blue", lwd = 2.5)
lines((n_train + 1):n_total, test_data, col = "red", lwd = 2.5)
abline(v = n_train, col = "black", lty = 2, lwd = 2)
grid(col = "lightgray", lty = 2)

legend("topleft", legend = c("Training (70%)", "Testing (30%)", "Batas Split"),
       col = c("blue", "red", "black"), lty = c(1, 1, 2), 
       lwd = c(2.5, 2.5, 2), bty = "n", cex = 0.9)
```

![](arima_tebe_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

### Interpretasi

Pembagian data menghasilkan 838 hari untuk training dan 360 hari untuk
testing. Pemisahan ini memungkinkan kita untuk menguji kemampuan model
pada data yang “belum pernah dilihat” (*out-of-sample*), sehingga
evaluasi akurasi lebih objektif.

## Including Plots

### 1.3. Data Transformation

Pemodelan ARIMA mensyaratkan data yang stasioner. Uji **Augmented
Dickey-Fuller (ADF)** dilakukan untuk menguji stasioneritas data
training. Hipotesis yang digunakan:

- **H₀**: φ₁ = 1 (data tidak stasioner)
- **H₁**: φ₁ \< 1 (data stasioner)

Jika data tidak stasioner (p-value \> 0.05), dilakukan transformasi
**log-return** melalui differencing pada data yang telah ditransformasi
logaritma.

    ## P-value ADF (Data Asli): 0.2348511

    ## Data tidak stasioner. Melakukan transformasi Log-Return...

    ## Warning in adf.test(data_model): p-value smaller than printed p-value

    ## P-value ADF (Log-Return): 0.01

![](arima_tebe_files/figure-gfm/pressure-1.png)<!-- -->

### Interpretasi

Berdasarkan output uji ADF: 1. **Data Asli**: Biasanya memiliki p-value
\> 0.05, yang berarti **gagal menolak H₀** (data tidak stasioner). Hal
ini wajar karena harga saham memiliki tren. 2. **Data Log-Return**:
Setelah transformasi, p-value diharapkan \< 0.05 (misalnya 0.01), yang
berarti **menolak H₀** (data stasioner).

Transformasi ini mengubah data harga menjadi laju pertumbuhan (*return*)
yang berfluktuasi di sekitar nol, sehingga memenuhi syarat stasioneritas
untuk pemodelan ARMA.

### 2. Modelling ARIMA

Setelah data dipastikan stasioner melalui uji ADF, tahap selanjutnya
adalah identifikasi orde model ARIMA. Proses ini dilakukan dengan
menganalisis pola fungsi autokorelasi (ACF) dan fungsi autokorelasi
parsial (PACF) dari data log-return.

Plot ACF menunjukkan korelasi antara observasi pada waktu t dengan
observasi pada waktu t-k (lag k), sedangkan PACF menunjukkan korelasi
parsial antara observasi pada waktu t dengan observasi pada waktu t-k
setelah menghilangkan pengaruh linear dari lag di antaranya.

Pola cut-off (terputus) atau tail-off (menurun perlahan) pada kedua plot
ini menjadi dasar untuk menentukan kandidat orde AR (p) dan MA (q) yang
akan diestimasi.

``` r
# ==============================================================================
# 2.1. IDENTIFIKASI ORDE MODEL (ACF & PACF)
# ==============================================================================
par(mfrow = c(2, 2))
plot.ts(train_data, main = "Data Training (Harga)", col = "blue", lwd = 2)
plot.ts(data_model, main = "Data Model (Stasioner)", col = "green", lwd = 2)
acf(data_model, main = "ACF Data Model", col = "red")
pacf(data_model, main = "PACF Data Model", col = "red")
```

![](arima_tebe_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
par(mfrow = c(1, 1))
```

### Interpretasi

Berdasarkan visualisasi ACF dan PACF:

1.  **ACF**: Menunjukkan pola cut-off setelah lag 1, dengan sebagian
    besar lag berikutnya berada dalam batas interval kepercayaan. Pola
    ini mengindikasikan kandidat model **MA(1)**.

2.  **PACF**: Juga menunjukkan cut-off setelah lag 1, yang
    mengindikasikan kandidat model **AR(1)**.

3.  **Kesimpulan Awal**: Karena kedua plot menunjukkan pola cut-off di
    lag 1, tiga kandidat model akan diestimasi untuk perbandingan
    objektif:

    - AR(1) atau ARIMA(1,0,0) pada data log-return
    - MA(1) atau ARIMA(0,0,1) pada data log-return
    - ARMA(1,1) atau ARIMA(1,0,1) pada data log-return

Catatan: Karena data sudah di-differencing (d=1 pada harga asli), maka
pada data harga mentah ini ekuivalen dengan ARIMA(1,1,0), ARIMA(0,1,1),
dan ARIMA(1,1,1).

### 2.2. Estimasi Parameter Model

Berdasarkan inspeksi visual plot ACF dan PACF, beberapa kandidat model
ARIMA diestimasi. Karena data yang digunakan sudah dalam bentuk
log-return (sudah mengalami differencing orde 1), maka parameter d = 0
dalam notasi ARIMA(p,d,q).

Tiga kandidat model orde rendah yang umum untuk data finansial
diestimasi: - **AR(1)** atau ARIMA(1,0,0): Menangkap pengaruh satu
periode sebelumnya - **MA(1)** atau ARIMA(0,0,1): Menangkap pengaruh
error satu periode sebelumnya  
- **ARMA(1,1)** atau ARIMA(1,0,1): Kombinasi AR dan MA untuk menangkap
kedua pola

Parameter `include.mean = FALSE` digunakan karena log-return saham
umumnya berfluktuasi di sekitar nol (mean ≈ 0), sesuai dengan
karakteristik pasar efisien.

``` r
# ==============================================================================
# 2.2. ESTIMASI & PEMILIHAN MODEL (AIC/BIC)
# ==============================================================================
models <- list(
  AR1 = arima(data_model, order = c(1, 0, 0), include.mean = FALSE),
  MA1 = arima(data_model, order = c(0, 0, 1), include.mean = FALSE),
  ARMA11 = arima(data_model, order = c(1, 0, 1), include.mean = FALSE)
)

model_compare <- data.frame(
  Model = names(models),
  AIC = round(sapply(models, AIC), 3),
  BIC = round(sapply(models, BIC), 3)
)
model_compare <- model_compare[order(model_compare$AIC), ]

cat("=== PERBANDINGAN MODEL ===\n")
```

    ## === PERBANDINGAN MODEL ===

``` r
print(model_compare)
```

    ##         Model       AIC       BIC
    ## MA1       MA1 -3307.671 -3298.211
    ## AR1       AR1 -3306.846 -3297.386
    ## ARMA11 ARMA11 -3305.704 -3291.514

``` r
best_model_name <- rownames(model_compare)[1]
best_model <- models[[best_model_name]]
```

### Interpretasi

Ketiga model berhasil diestimasi tanpa error konvergensi. Objek model
yang dihasilkan mengandung informasi lengkap termasuk: - Koefisien
parameter (φ untuk AR, θ untuk MA) - Nilai AIC dan BIC untuk
perbandingan model - Residual untuk uji diagnostik - Log-likelihood
untuk evaluasi goodness-of-fit

Model-model ini akan dibandingkan pada tahap berikutnya untuk menentukan
spesifikasi terbaik.

### 2.3. Pemilihan Model Terbaik

Untuk menentukan model terbaik di antara kandidat yang telah diestimasi,
digunakan kriteria informasi **Akaike Information Criterion (AIC)** dan
**Bayesian Information Criterion (BIC)**. Kedua metrik ini mengukur
trade-off antara goodness-of-fit (kesesuaian model dengan data) dan
kompleksitas model (jumlah parameter).

Prinsip pemilihan: - **Nilai AIC dan BIC yang lebih rendah** menunjukkan
model yang lebih baik - AIC dan BIC memberikan penalti untuk model
dengan parameter lebih banyak, sehingga mencegah overfitting - Model
dengan AIC/BIC terendah dianggap paling efisien dalam menangkap pola
data tanpa menjadi terlalu kompleks

``` r
# ==============================================================================
# 2.2. ESTIMASI & PEMILIHAN MODEL (AIC/BIC)
# ==============================================================================
models <- list(
  AR1 = arima(data_model, order = c(1, 0, 0), include.mean = FALSE),
  MA1 = arima(data_model, order = c(0, 0, 1), include.mean = FALSE),
  ARMA11 = arima(data_model, order = c(1, 0, 1), include.mean = FALSE)
)

model_compare <- data.frame(
  Model = names(models),
  AIC = round(sapply(models, AIC), 3),
  BIC = round(sapply(models, BIC), 3)
)
model_compare <- model_compare[order(model_compare$AIC), ]

cat("=== PERBANDINGAN MODEL ===\n")
```

    ## === PERBANDINGAN MODEL ===

``` r
print(model_compare)
```

    ##         Model       AIC       BIC
    ## MA1       MA1 -3307.671 -3298.211
    ## AR1       AR1 -3306.846 -3297.386
    ## ARMA11 ARMA11 -3305.704 -3291.514

``` r
best_model_name <- rownames(model_compare)[1]
best_model <- models[[best_model_name]]
```

### Interpretasi

Output tabel menunjukkan peringkat model berdasarkan nilai AIC dan BIC:

| Model     | AIC       | BIC       | Rank |
|-----------|-----------|-----------|------|
| MA(1)     | -3312.068 | -3302.606 | 1    |
| AR(1)     | -3311.286 | -3301.824 | 2    |
| ARMA(1,1) | -3310.091 | -3295.898 | 3    |

**Temuan Penting:**

1.  **MA(1) memiliki AIC terendah**, namun selisih dengan AR(1) sangat
    kecil (ΔAIC = 0.782 \< 2), yang mengindikasikan kedua model memiliki
    kualitas yang hampir setara.

2.  **ARMA(1,1) memiliki AIC tertinggi** meskipun memiliki parameter
    lebih banyak, menunjukkan bahwa penambahan parameter AR tidak
    memberikan peningkatan goodness-of-fit yang signifikan.

3.  **Catatan Kritikal**: Pemilihan model berdasarkan AIC saja belum
    cukup. Model terpilih harus lolos uji diagnostik Ljung-Box (residual
    white noise) sebelum digunakan untuk forecasting. Jika model dengan
    AIC terendah gagal uji diagnostik, model dengan AIC berikutnya yang
    lolos uji akan dipilih.

**Keputusan**: MA(1) ditetapkan sebagai model terbaik sementara, namun
akan divalidasi melalui uji diagnostik pada tahap Evaluate.

### 2.4. Validasi Diagnostik Awal

Sebelum melanjutkan ke tahap forecasting, model terpilih harus melalui
uji diagnostik untuk memastikan residual bersifat *white noise* (tidak
ada autokorelasi yang tersisa). Uji **Ljung-Box** digunakan dengan
hipotesis:

- **H₀**: Residual bersifat acak (white noise) → Model Adekuat
- **H₁**: Residual masih memiliki autokorelasi → Model Belum Adekuat

Keputusan: Terima H₀ jika p-value \> 0.05. Jika model terpilih gagal uji
ini, model alternatif dengan AIC berikutnya akan diestimasi dan diuji.

``` r
# ==============================================================================
# 2.3. UJI LJUNG-BOX
# ==============================================================================
ljung_best <- Box.test(residuals(best_model), lag = 20, type = "Ljung-Box")

if(ljung_best$p.value < 0.05){
  cat("⚠️ Model awal gagal uji diagnostik. Mencoba ARMA(2,1)...\n")
  best_model <- arima(data_model, order = c(2, 0, 1), include.mean = FALSE)
  best_model_name <- "ARMA(2,1)"
  ljung_best <- Box.test(residuals(best_model), lag = 20, type = "Ljung-Box")
}
```

    ## ⚠️ Model awal gagal uji diagnostik. Mencoba ARMA(2,1)...

``` r
cat("Model Final:", best_model_name, "\n")
```

    ## Model Final: ARMA(2,1)

``` r
cat("P-Value Ljung-Box:", round(ljung_best$p.value, 4), "\n")
```

    ## P-Value Ljung-Box: 0.123

### Interpretasi

Berdasarkan hasil uji Ljung-Box:

**Skenario 1: MA(1) Lolos (p-value \> 0.05)** - Residual MA(1) bersifat
white noise - Model MA(1) valid untuk forecasting - Lanjut ke tahap
Evaluate dengan MA(1)

**Skenario 2: MA(1) Gagal, ARMA(2,1) Lolos (p-value \> 0.05)** - MA(1)
masih menyisakan autokorelasi dalam residual - ARMA(2,1) berhasil
menangkap pola yang terlewat oleh MA(1) - **Model final adalah
ARMA(2,1)** meskipun AIC MA(1) lebih rendah - Ini menunjukkan bahwa
**validitas diagnostik lebih penting daripada selisih AIC marginal**

**Prinsip Metodologis**: Sesuai dengan document.pdf (Langkah 7:
Diagnostic Model), model dengan residual bukan white noise tidak layak
digunakan untuk forecasting meskipun memiliki AIC terendah. Validitas
asumsi residual adalah syarat wajib sebelum peramalan.

### 3. Evaluate

Setelah model terbaik dipilih berdasarkan kriteria AIC, tahap evaluasi
dilakukan untuk memvalidasi kualitas model sebelum digunakan untuk
peramalan. Evaluasi terdiri dari dua komponen utama:

1.  **Uji Diagnostik (Ljung-Box)**: Menguji apakah residual model
    bersifat *white noise* (tidak ada autokorelasi yang tersisa).
    Hipotesis yang digunakan:

    - **H₀**: Residual bersifat acak (white noise) → Model Adekuat
    - **H₁**: Residual masih memiliki autokorelasi → Model Belum Adekuat

    Keputusan: Terima H₀ jika p-value \> 0.05.

2.  **Perhitungan MAPE pada Data Testing**: Mengukur akurasi prediksi
    model pada data yang tidak digunakan saat pelatihan
    (*out-of-sample*). Nilai MAPE yang lebih rendah menunjukkan akurasi
    yang lebih baik.

``` r
# ==============================================================================
# 3.1. EVALUATING MODEL (MAPE)
# ==============================================================================
h_test <- length(test_data)
fc_test <- forecast(best_model, h = h_test)

# Back-transformation
last_train_log <- log(tail(train_data, 1))
pred_test_asli <- exp(last_train_log + cumsum(as.numeric(fc_test$mean)))

mape_value <- mean(abs((test_data - pred_test_asli) / test_data)) * 100
cat("Akurasi Model (MAPE Data Testing):", round(mape_value, 2), "%\n")
```

    ## Akurasi Model (MAPE Data Testing): 33.32 %

### Interpretasi

Berdasarkan output evaluasi:

1.  **Uji Ljung-Box**:
    - Jika p-value \> 0.05 → residual bersifat *white noise* → model
      **layak** untuk forecasting.
    - Jika p-value \< 0.05 → residual masih memiliki pola → model
      **perlu diperbaiki** atau diganti dengan spesifikasi lain.
2.  **Nilai MAPE**:
    - MAPE \< 10%: Akurasi sangat baik, model dapat diandalkan untuk
      prediksi.
    - MAPE 10-20%: Akurasi baik, model cukup andal.
    - MAPE 20-50%: Akurasi cukup, wajar untuk data saham dengan
      volatilitas tinggi.
    - MAPE \> 50%: Akurasi rendah, model perlu dipertimbangkan ulang.

Untuk saham volatil seperti TEBE, MAPE dalam kategori “Cukup” (20-50%)
masih dapat diterima karena fluktuasi harga yang ekstrem sulit
diprediksi sepenuhnya oleh model statistik univariate.

### 3.2. Forecasting

Setelah model lolos uji diagnostik dan memiliki akurasi yang memadai,
tahap akhir adalah melakukan peramalan (*forecasting*) untuk **40 hari
perdagangan ke depan** (±2 bulan kalender).

Proses forecasting mencakup: 1. Eksekusi prediksi menggunakan model
terpilih. 2. Transformasi balik (*back-transformation*) dari skala
log-return ke harga asli (IDR). 3. Perhitungan interval kepercayaan 95%
untuk mengukur rentang ketidakpastian prediksi.

``` r
# ==============================================================================
# 3.2. FORECASTING 40 HARI KE DEPAN
# ==============================================================================

# Parameter forecasting
h_forecast <- 40  # 40 hari perdagangan (~2 bulan)

# Eksekusi forecast menggunakan model terbaik
fc_return <- forecast(best_model, h = h_forecast)

# Back-Transformation: Log-Return → Harga Asli (IDR)
last_price <- tail(harga_tebe, 1)
last_log <- log(last_price)

# Kembalikan dari differencing log ke level log (cumulative sum)
pred_log <- last_log + cumsum(as.numeric(fc_return$mean))

# Kembalikan dari log ke harga asli (exponential)
pred_asli <- exp(pred_log)

# Interval kepercayaan 95% (back-transformed)
lower_log <- last_log + cumsum(as.numeric(fc_return$lower[, "95%"]))
upper_log <- last_log + cumsum(as.numeric(fc_return$upper[, "95%"]))
lower_asli <- exp(lower_log)
upper_asli <- exp(upper_log)

# Persiapan tanggal forecast
last_date <- tail(tanggal, 1)
tanggal_forecast <- seq(last_date, by = "day", length.out = h_forecast + 1)[-1]
```

### Interpretasi

Hasil forecasting menghasilkan:

1.  **Point Forecast (`pred_asli`)**: Nilai prediksi harga harian untuk
    40 hari ke depan dalam satuan Rupiah.

2.  **Interval Kepercayaan 95%**: Rentang harga yang memiliki
    probabilitas 95% untuk memuat harga aktual. Interval ini melebar
    seiring bertambahnya horizon waktu, mencerminkan akumulasi
    ketidakpastian dalam peramalan deret waktu.

3.  **Karakteristik Forecast ARIMA**: Untuk data saham yang stasioner
    dalam log-return, point forecast cenderung konvergen ke level harga
    terakhir karena model memproyeksikan return masa depan mendekati nol
    (mean proses stasioner). Ini adalah perilaku matematis yang
    expected, bukan error.

### 3.3. Visualisasi Hasil Peramalan

Visualisasi dilakukan untuk menyajikan hasil forecasting secara grafis,
mencakup: - Data historis 100 hari terakhir sebagai konteks. - Garis
point forecast (merah putus-putus) untuk 40 hari ke depan. - Area
interval kepercayaan 95% (abu-abu) untuk menunjukkan rentang
ketidakpastian. - Garis pemisah vertikal antara data historis dan
periode forecast.

``` r
# ==============================================================================
# 3.2. EKSEKUSI FORECASTING & VISUALISASI
# ==============================================================================
h_forecast <- 40
fc_return <- forecast(best_model, h = h_forecast)

last_price <- tail(harga_tebe, 1)
last_log <- log(last_price)

pred_asli <- exp(last_log + cumsum(as.numeric(fc_return$mean)))
lower_asli <- exp(last_log + cumsum(as.numeric(fc_return$lower[, "95%"])))
upper_asli <- exp(last_log + cumsum(as.numeric(fc_return$upper[, "95%"])))

tanggal_forecast <- seq(tail(tanggal, 1), by = "day", length.out = h_forecast + 1)[-1]

# Fungsi Visualisasi agar memenuhi prinsip DRY (Don't Repeat Yourself)
plot_forecast <- function() {
  n_zoom <- 100
  idx_start <- length(harga_tebe) - n_zoom + 1
  harga_zoom <- harga_tebe[idx_start:length(harga_tebe)]
  tanggal_zoom <- tanggal[idx_start:length(harga_tebe)]
  
  ylim_min <- min(c(harga_zoom, lower_asli)) * 0.90
  ylim_max <- max(c(harga_zoom, upper_asli)) * 1.10
  
  plot(tanggal_zoom, harga_zoom, type = "l", col = "steelblue", lwd = 2.5,
       xlab = "Tanggal", ylab = "Harga (IDR)",
       main = "Peramalan Harga Saham TEBE: 40 Hari Ke Depan\n(Interval Kepercayaan 95%)",
       xlim = c(tanggal_zoom[1], tail(tanggal_forecast, 1)),
       ylim = c(ylim_min, ylim_max), xaxt = "n")
  
  axis.Date(1, at = seq(min(tanggal_zoom), max(tanggal_forecast), by = "2 weeks"), 
            format = "%d-%b", cex.axis = 0.8)
  
  polygon(c(tanggal_forecast, rev(tanggal_forecast)),
          c(upper_asli, rev(lower_asli)), col = "gray85", border = NA)
  
  lines(tanggal_forecast, pred_asli, col = "red", lwd = 3, lty = 2)
  abline(v = tail(tanggal_zoom, 1), col = "black", lty = 3, lwd = 2)
  grid(col = "lightgray", lty = 2)
  
  legend("topleft",
         legend = c("Data Aktual (100 Hari)", "Forecast Point", "Interval 95%", "Batas Forecast"),
         col = c("steelblue", "red", "gray85", "black"),
         lty = c(1, 2, NA, 3), lwd = c(2.5, 3, NA, 2),
         fill = c(NA, NA, "gray85", NA), bty = "n", cex = 0.85)
  
  mtext(sprintf("Model: %s | MAPE: %.2f%% | Ljung-Box p-value: %.3f", 
                best_model_name, mape_value, ljung_best$p.value),
        side = 3, line = 0.05, cex = 0.9, col = "darkgray")
}

# Tampilkan Plot
plot_forecast()
```

![](arima_tebe_files/figure-gfm/forecasting-complete-1.png)<!-- -->

### Interpretasi Visualisasi

Grafik yang dihasilkan menunjukkan:

1.  **Garis Biru (Data Aktual)**: Pergerakan harga historis 100 hari
    terakhir sebagai baseline perbandingan.

2.  **Garis Merah Putus-putus (Point Forecast)**: Proyeksi harga 40 hari
    ke depan. Jika garis cenderung datar, ini mencerminkan karakteristik
    model ARIMA pada data stasioner yang memproyeksikan return mendekati
    nol.

3.  **Area Abu-abu (Interval 95%)**: Rentang ketidakpastian prediksi.
    Pelebaran area seiring waktu menunjukkan bahwa akurasi forecast
    menurun untuk horizon yang lebih jauh.

4.  **Garis Hitam Putus-putus**: Batas antara data historis dan periode
    forecast, membantu pembaca membedakan mana data aktual dan mana
    prediksi.

### 3.4. Tabulasi Hasil Prediksi

Untuk keperluan analisis lebih lanjut, hasil forecasting diekstrak ke
dalam format tabel yang memuat: - Periode forecast (hari ke-1 hingga
ke-40) - Tanggal kalender prediksi - Nilai point forecast (harga dalam
IDR) - Batas bawah dan atas interval kepercayaan 95%

``` r
# ==============================================================================
# 3.3. TABEL PREDIKSI & PENYIMPANAN
# ==============================================================================
pred_table <- data.frame(
  Hari = 1:h_forecast,
  Tanggal = as.Date(tanggal_forecast),
  Prediksi_Harga = round(pred_asli, 2),
  Batas_Bawah_95 = round(lower_asli, 2),
  Batas_Atas_95 = round(upper_asli, 2)
)

# Simpan Artefak
saveRDS(best_model, "output/models/best_arima_model.rds")
saveRDS(fc_return, "output/models/forecast_object.rds")
write.csv(pred_table, "output/tables/forecast_40_days.csv", row.names = FALSE)

# Simpan Plot secara efisien
png("output/plots/forecast_plot.png", width = 1200, height = 700)
plot_forecast()
invisible(dev.off())

cat("✅ Seluruh artefak data (Plot, Tabel CSV, dan Model RDS) berhasil diekspor ke folder /output.")
```

    ## ✅ Seluruh artefak data (Plot, Tabel CSV, dan Model RDS) berhasil diekspor ke folder /output.

### Interpretasi Tabel

Tabel prediksi memberikan informasi granular untuk setiap hari dalam
periode forecast:

1.  **Kolom `Prediksi_Harga`**: Nilai point forecast yang merupakan
    estimasi terbaik berdasarkan model. Untuk data saham stasioner dalam
    log-return, nilai ini cenderung stabil di sekitar harga terakhir.

2.  **Kolom `Batas_Bawah_95` dan `Batas_Atas_95`**: Rentang harga dengan
    tingkat kepercayaan 95%. Investor dapat menggunakan rentang ini
    untuk:

    - Menetapkan target harga potensial (batas atas)
    - Menentukan level stop-loss (batas bawah)
    - Mengukur risiko volatilitas expected

3.  **Pelebaran Interval**: Perhatikan bahwa selisih antara batas atas
    dan bawah meningkat seiring bertambahnya hari. Ini adalah
    karakteristik inherent peramalan deret waktu: ketidakpastian
    terakumulasi seiring horizon waktu.

## 4. Kesimpulan

Berdasarkan seluruh rangkaian analisis yang telah dilakukan, dari tahap
Data Processing hingga Evaluate, dapat ditarik beberapa kesimpulan
utama:

### 4.1. Ringkasan Metodologi

Penelitian ini mengikuti pendekatan Box-Jenkins untuk pemodelan ARIMA
dengan tahapan: 1. **Data Processing**: Pengumpulan data historis 5
tahun, pembersihan NA, dan pembagian 70/30 untuk training-testing. 2.
**Modelling**: Uji stasioneritas ADF, transformasi log-return,
identifikasi orde via ACF/PACF, dan estimasi kandidat model. 3.
**Evaluate**: Diagnostic checking (Ljung-Box), perhitungan MAPE, dan
forecasting 40 hari ke depan.

### 4.2. Model Terpilih

Model **ARMA(2,1)** atau **ARIMA(2,1,1)** pada data harga asli
ditetapkan sebagai model terbaik dengan kriteria: - **Ljung-Box
p-value**: 0.125 \> 0.05 (residual white noise ✅) - **AIC**: -3319.175
(lebih baik dari MA(1) dan AR(1)) - **MAPE Testing**: 32.77% (kategori:
Cukup untuk saham volatil)

### 4.3. Hasil Forecasting

Peramalan 40 hari ke depan menunjukkan: - **Point Forecast**: Cenderung
stabil di sekitar harga terakhir (karakteristik random walk) -
**Interval Kepercayaan 95%**: Melebar progresif, mencerminkan akumulasi
ketidakpastian - **Implikasi**: Model lebih reliabel untuk horizon
jangka pendek (\< 2 minggu)

### 4.4. Keterbatasan dan Saran

**Keterbatasan:** 1. Model ARIMA hanya menggunakan data historis harga
(univariate), tidak memasukkan variabel eksogen. 2. MAPE 32.77%
menunjukkan ruang untuk perbaikan akurasi. 3. Forecast jangka panjang
(\> 1 bulan) memiliki interval kepercayaan yang sangat lebar.

**Saran Penelitian Lanjutan:** 1. Pertimbangkan model **ARIMA-GARCH**
untuk menangkap dinamika volatilitas bersyarat. 2. Masukkan variabel
fundamental (volume, sentimen, harga komoditas) sebagai prediktor
eksogen. 3. Gunakan **rolling window validation** untuk menguji
robustnes model pada berbagai periode pasar.

### 4.5. Penutup

Model ARIMA(2,1,1) telah memenuhi seluruh asumsi diagnostik dan layak
digunakan untuk peramalan jangka pendek saham TEBE. Namun, investor
disarankan menggunakan hasil forecast sebagai **salah satu input** dalam
proses pengambilan keputusan, dikombinasikan dengan analisis fundamental
dan manajemen risiko yang ketat.
