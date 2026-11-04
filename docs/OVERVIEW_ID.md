# Ringkasan proyek

AI Operations Assistant adalah asisten operasional untuk perusahaan kecil, dibuat dengan Python untuk mempelajari dan menunjukkan kemampuan AI engineering. Asisten ini menjawab pertanyaan dari dokumen beserta sumbernya, mencari data pesanan lewat agent, memastikan setiap perubahan disetujui manusia, dan mengukur kualitas retrieval-nya sendiri.

> **Ini proyek pembelajaran dengan data sintetis.** Proyek ini belum pernah di-deploy ke produksi, dan belum pernah dijalankan dengan LLM sungguhan. Lihat bagian Keterbatasan.

## Masalah yang diselesaikan

Staf perusahaan kecil sering menghabiskan waktu untuk mencari kebijakan, mengecek status pesanan, dan menjawab tiket yang mirip. Model AI bisa membantu, tetapi ada dua risiko: model bisa mengarang jawaban, dan model bisa dimanipulasi oleh teks yang dibacanya. Proyek ini fokus pada dua risiko itu.

## Fitur utama

| Fitur | Cara kerja |
|---|---|
| **Menjawab dari dokumen (RAG)** | Mencari teks yang relevan, menjawab hanya dari teks itu, dan menampilkan sumbernya. Kalau tidak ada yang relevan, sistem menyatakannya tanpa memanggil model |
| **Agent dengan tool** | Loop yang ditulis sendiri untuk mencari dokumen dan membaca pesanan, dengan batas langkah dan waktu |
| **Persetujuan manusia** | Model hanya bisa mengusulkan perubahan. Perubahan hanya berjalan setelah disetujui orang, dan tidak ada tool yang memungkinkan model menyetujui |
| **Workflow tiket** | Mengklasifikasi tiket, mencari pesanan, menyusun draft balasan, lalu menunggu persetujuan sebelum mengirim. Pengulangan tidak pernah mengirim dua kali |
| **Komponen yang bisa diganti** | Provider LLM, vector store, penyimpanan data, dan layanan pesan masing-masing berada di balik interface |
| **Evaluasi** | Kumpulan pertanyaan tetap mengukur retrieval, dan CI gagal kalau skornya turun |
| **Antarmuka web** | Halaman sederhana untuk tiap fitur, tanpa langkah build, disajikan oleh API |

## Cara mencoba

Butuh Python 3.12 dan Git. API key tidak diperlukan, karena ada mock provider sebagai pengganti model. Di Windows (PowerShell):

```powershell
git clone https://github.com/maeshataruna99-lgtm/core-ai-automation.git
cd core-ai-automation
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/demo.py
```

Perintah untuk Linux ada di `README.md`. Demo ini menyalakan API, menjalankan tiap fitur, memeriksa setiap hasil, lalu mematikan API.

Mock provider hanya mengulang pertanyaan dan tidak menjawabnya. Jadi demo menunjukkan retrieval, tool, persetujuan, dan workflow, tetapi bukan kualitas jawaban.

## Antarmuka web

Untuk memakainya lewat browser, jalankan perintah ini dari folder utama repo, lalu buka `http://127.0.0.1:8000`:

```powershell
$env:AI_OPS_LLM_PROVIDER = "mock"
python -m ai_ops
```

Rekaman pemakaiannya ada di `docs/demo/ui-demo.mp4`, dengan keterangan berbahasa Inggris.

## Hasil evaluasi

| Metrik, pada 24 pertanyaan yang bisa dijawab dan 8 yang tidak bisa | Hasil |
|---|---|
| Dokumen yang benar terambil (`hit_rate`) | 0,958 |
| Teks yang terambil memuat fakta yang diharapkan (`context_fact_recall`) | 0,958 |
| Pertanyaan yang tidak bisa dijawab mendapat "tidak ditemukan" (`not_found_accuracy`) | 0,500 |
| Kualitas jawaban, sitasi, dan ketahanan terhadap prompt injection | Belum diukur, butuh model sungguhan |

**Angka ini menggambarkan retrieval dengan mock yang hanya mencocokkan kata persis, pada data sintetis. Ini bukan akurasi model AI.** Pengaturannya juga dioptimasi pada pertanyaan yang sama, jadi hasil pada pertanyaan baru kemungkinan lebih rendah. Penjelasan lengkapnya ada di `docs/EVALUATION.md`.

## Keterbatasan

- **Belum memakai LLM sungguhan.** Provider OpenAI-compatible dan tool calling baru dites dengan respons simulasi. Cara model asli menjawab, memilih tool, atau bereaksi terhadap prompt injection belum terverifikasi.
- **Adapter pesan HTTP belum dites ke vendor asli.** Vendor sungguhan kemungkinan memerlukan penyesuaian.
- **Tidak ada login.** Siapa pun yang bisa mengakses API bisa membaca pesanan dan menyetujui aksi, jadi API hanya terikat ke localhost. Jangan membukanya ke jaringan.
- **Sebagian data hilang saat restart.** Pesanan tersimpan di SQLite, tetapi persetujuan, progres workflow, dan vector store ada di memori.
- **Hanya untuk satu pengguna dan skala kecil.**
- **Log bisa memuat data pribadi,** karena input tool dicatat sebagai jejak audit.
- **Antarmuka webnya sengaja sederhana.** Riwayat chat dan diagnostics hilang saat halaman dimuat ulang.
- **Belum ada Redis cache.** Ini opsional, dan belum dibutuhkan.

Daftar lengkap dan dokumentasi lainnya ada di `README.md`, yang ditulis dalam bahasa Inggris.
