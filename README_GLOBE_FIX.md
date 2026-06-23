# Globe Fix

Versi ini memakai Plotly native globe (`Scattergeo` + `orthographic projection`), bukan HTML/CSS globe.

Jika globe tidak muncul:
1. Pastikan `plotly` ada di `requirements.txt`.
2. Upload file di root repository: `app.py`, `requirements.txt`, `.streamlit/config.toml`.
3. Di Streamlit Cloud, klik **Reboot app** setelah commit.
4. Clear cache dari app menu jika masih menampilkan versi lama.


## No Sidebar Version

Versi ini menyembunyikan seluruh sidebar. Dataset dimuat otomatis. Jika Kaggle Secrets belum ada, dashboard memakai demo data secara otomatis.
