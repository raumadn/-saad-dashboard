# SAAD Dealer | Streamlit Cloud Ready Dashboard

ZIP ini sudah diperbaiki untuk deploy di Streamlit Cloud.

## Penting

API key Kaggle **tidak dimasukkan** ke ZIP ini supaya tidak bocor saat project di-upload ke GitHub. Dashboard tetap siap memakai Kaggle API. Credential dimasukkan lewat **Streamlit Cloud Secrets**, bukan file repository.

## Struktur

```text
saad_dealer_streamlit_cloud_ready/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── run_local.bat
├── run_local.sh
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

## Deploy ke Streamlit Cloud

1. Upload isi folder ini ke GitHub.
2. Buka Streamlit Cloud.
3. Create app.
4. Pilih repository.
5. Main file path:

```text
app.py
```

6. Buka **Advanced settings / Secrets**.
7. Paste format ini:

```toml
[kaggle]
username = "djoenorris"
key = "PASTE_KAGGLE_API_KEY_BARU_DI_SINI"
```

8. Klik Deploy.
9. Kalau mengubah Secrets setelah deploy, klik **Reboot app**.

## Local testing

Buat file lokal:

```text
.streamlit/secrets.toml
```

Isi:

```toml
[kaggle]
username = "djoenorris"
key = "PASTE_KAGGLE_API_KEY_BARU_DI_SINI"
```

Lalu jalankan:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## File yang tidak boleh di-upload ke GitHub

```text
kaggle.json
.streamlit/secrets.toml
secrets.toml
data/
```

File-file tersebut sudah masuk `.gitignore`.

## Demo mode

Jika Kaggle Secrets belum terdeteksi, dashboard otomatis membuka **demo data** agar tampilan tidak langsung error. Setelah Secrets benar, matikan toggle **Use demo data** di sidebar untuk memakai data Kaggle asli.
