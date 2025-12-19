# AI-Emotion-Project
AI Emotion Project adalah proyek kecerdasan buatan yang bertujuan untuk mendeteksi emosi pada kalimat berbahasa Inggris menggunakan pendekatan machine learning. Proyek ini berfokus pada pengujian dan perbandingan lima algoritma machine learning untuk mengetahui model mana yang paling efektif dalam mengklasifikasikan emosi dari teks.

1. Support Vector Machine (SVM)
<img width="598" height="301" alt="image" src="https://github.com/user-attachments/assets/06aa4bee-d296-4d48-a449-bdad48dbc1cd" />
Model SVM menghasilkan akurasi sebesar 0.8885. Setiap emosi diklasifikasikan dengan cukup baik, ditunjukkan oleh nilai precision dan recall yang tinggi. Hal ini menunjukkan bahwa SVM mampu mengenali pola dari fitur TF-IDF dengan baik. Namun, performanya masih sedikit di bawah Random Forest.

2. K-Nearest Neighbor (KNN)
<img width="542" height="287" alt="image" src="https://github.com/user-attachments/assets/05d030b2-c45f-48a7-83d2-1e6a0a5f8f73" />
Model Model KNN menghasilkan akurasi sebesar 0.8200. Nilai precision dan recall pada kelas tertentu seperti cinta (love) dan terkejut (surprise) cenderung lebih rendah dibanding model lain. Hal ini menunjukkan bahwa KNN kurang efektif untuk dataset teks berdimensi tinggi (TF-IDF), karena model ini bergantung pada jarak antar vektor, sehingga performanya menurun ketika jumlah fitur sangat banyak.

3. Random Forest
<img width="595" height="312" alt="image" src="https://github.com/user-attachments/assets/5a9f5924-892c-48be-bd84-03b2e1993c57" />
Model Random Forest menghasilkan akurasi tertinggi, yaitu 0.8895. Model ini memiliki nilai precision, recall, dan f1-score yang stabil di semua kelas emosi. Hal ini menunjukkan bahwa Random Forest mampu mempelajari pola non-linear dan memanfaatkan ensemble decision tree, sehingga memberikan hasil yang paling optimal dibandingkan model lain.

4. Logistic Regression
<img width="590" height="310" alt="image" src="https://github.com/user-attachments/assets/86633005-a0ef-416a-9583-cca2b3e79c0f" />
Model Logistic Regression menghasilkan akurasi sebesar 0.8690. Meskipun performanya cukup baik, terutama untuk kelas sadness dan joy, namun ada kategori yang kurang stabil, misalnya kelas surprise. Model ini masih kalah dibanding SVM dan Random Forest dalam hal fleksibilitas menangkap variasi data.

5. Multinomial Naïve Bayes
 <img width="595" height="308" alt="image" src="https://github.com/user-attachments/assets/b81c6c66-2e21-4df8-a6de-eced0c98025e" />
Model Multinomial NB menghasilkan akurasi 0.7615, yang merupakan nilai terendah dibandingkan model lainnya. Meskipun Naïve Bayes bekerja cukup baik untuk data teks, model ini mengasumsikan independensi antar fitur, sehingga kurang optimal saat pola antar kata saling terkait (misalnya frasa emosional).
