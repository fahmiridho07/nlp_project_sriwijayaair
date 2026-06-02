# Portfolio Evidence Pack

Gunakan file PNG di folder ini sebagai visual evidence untuk website portfolio.

## Recommended Website Order

1. `01_case_study_overview.png` - ringkasan problem, dataset, model terbaik, dan skor utama.
2. `02_data_label_evidence.png` - bukti sumber label dari rating Google Play.
3. `03_model_performance_evidence.png` - perbandingan model dan F1-score per kelas.
4. `04_confusion_matrix_evidence.png` - bukti evaluasi prediksi pada test set.
5. `05_user_pain_points.png` - insight kata/isu yang sering muncul di review negatif.
6. `06_sentiment_trend_by_month.png` - gambaran potensi monitoring sentimen dari waktu ke waktu.
7. `07_review_examples.png` - contoh review yang diklasifikasikan model.

## Copy for Portfolio Website

Saya membangun pipeline NLP end-to-end untuk menganalisis ulasan aplikasi Sriwijaya Air Mobile di Google Play Store. Data review diproses melalui cleaning dan feature engineering berbasis TF-IDF n-gram, lalu dibandingkan menggunakan Logistic Regression, Linear SVM, dan Naive Bayes. Model terbaik adalah Logistic Regression dengan accuracy 0.8523, macro F1 0.6363, dan weighted F1 0.8450.

Project ini juga menghasilkan insight produk: review negatif banyak menyinggung promo, tiket, fitur member, loading, aplikasi sulit dibuka, dan proses check-in. Kelas neutral masih menjadi tantangan karena jumlah datanya kecil dan rating 3 sering berisi campuran keluhan serta apresiasi.
