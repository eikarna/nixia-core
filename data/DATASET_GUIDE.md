# Nixia Dataset Guide

Tujuan dataset Nixia adalah chat Bahasa Indonesia kasual/roleplay ringan yang tetap bersih, legal, dan cocok untuk model kecil on-device.

## Sumber yang sudah dicek

| ID | Sumber | License | Default | Catatan |
|---|---|---:|---:|---|
| `nixia_seed` | `data/sample_corpus.txt` | project-local | ya | Seed manual gaya Nixia |
| `lorthgyu_indonesian_chat` | HF `LorthGyu/indonesian-chat` | MIT | ya | 200 multi-turn chat Indo |
| `lorthgyu_indonesian_qa` | HF `LorthGyu/indonesian-qa` | MIT | tidak | QA umum kecil sebagai suplemen pengetahuan ringan |
| `w11wo_twitter_indonesia_sarcastic` | HF `w11wo/twitter_indonesia_sarcastic` | Apache-2.0 | tidak | Social-style seed dari tweet termask; wajib filter/spot-check karena raw social media noisy/politis |
| `suryaadhi_ppmb_qa_id` | HF `suryaadhi/ppmb-qa-dataset` | MIT | tidak | QA helpdesk Indonesia; jawaban dipendekkan builder karena domain spesifik dan sering panjang |
| `gabrielb_python_qa` | HF `gabrielb/QA-Python-Programming-Indonesia` | Apache-2.0 | tidak | QA teknis Python; pakai sebagai suplemen kecil agar Nixia tidak berubah jadi coding assistant |
| `indonlp_cendol_chat_v2` | HF `indonlp/cendol_collection_v2` | Apache-2.0 | tidak | Besar; cocok untuk pretraining/instruction, bukan default chat kasual |
| `seacrowd_seadialogues` | HF `SEACrowd/SEADialogues` | CC-BY-SA-4.0 | perlu `--allow-sharealike` | Multi-turn cultural dialogue; bagus untuk lokal/kultural |
| `indonlp_nusax_mt` | HF `indonlp/NusaX-MT` | CC-BY-SA-4.0 | tidak | Bagus untuk tokenizer/dialek, bukan chat utama |
| `noura_roleplay_chat` | HF `Nourivex/Noura-Roleplay-Chat` | CC-BY-NC-SA-4.0 | tidak | Non-komersial; pakai hanya jika sadar batasan license |

`indolem/indobert-base-uncased` berguna sebagai referensi ekosistem NLP Indonesia, tetapi itu model BERT encoder, bukan sumber chat untuk causal LM.

## Build corpus kecil untuk development

```bash
python tools/build_dataset.py \
  --max-rows-per-source 200 \
  --synthesize 500 \
  --output data/curated/train_corpus.txt \
  --valid-output data/curated/valid_corpus.txt
```

Default hanya memakai sumber chat kasual/permissive (`nixia_seed`, `lorthgyu_indonesian_chat`) plus synthetic bersih jika diminta.

Prinsip base corpus:

- netral-kasual, bukan meme/persona tertentu,
- tidak menyuntik dialek/slang berat secara acak,
- gaya Jawa/Sunda/slang spesifik masuk style pack atau fine-tuning terpisah,
- tujuan: mengurangi persona leakage dan context rot.

## Target kualitas praktis

Anggap dataset punya tiga level:

1. **Smoke/dev**: cukup untuk cek pipeline jalan. Boleh kecil, tapi hasil model tidak dinilai serius.
2. **Fine-tune kecil**: minimal sekitar 1k+ dialog train dan 100+ dialog valid, format bersih, tanpa overlap train/valid.
3. **Training lebih serius**: target awal 5k-20k dialog real/curated, valid 500-2k dialog, synthetic maksimal 30%.

Untuk Nixia, contoh dialog bagus adalah multi-turn pendek yang natural:

```text
<user> aku pengen cerita tapi takut ganggu
<char> kamu gak ganggu kok. cerita pelan-pelan aja, aku dengerin.
<user> temenku tiba-tiba cuek, aku jadi kepikiran
<char> wajar kalau kepikiran. terakhir kalian ngobrol soal apa?
```

Checklist dialog yang layak masuk train/valid:

- role jelas dan konsisten: `<user>` lalu `<char>`, boleh 2-10 turn,
- respons nyambung dengan konteks sebelumnya,
- gaya kasual Indonesia sesuai target, tidak terlalu formal/ensiklopedis,
- bukan template yang sama berulang-ulang,
- tidak ada URL, nomor telepon, email, handle, secret, atau data pribadi,
- tidak ada prompt/model artifact seperti "sebagai AI/model bahasa",
- aman secara lisensi: buatan sendiri, public-domain/permissive, atau sumber yang memang diizinkan.

Jika memakai chat pribadi, simpan mentahnya di `data/private/` atau `data/raw/` karena folder itu di-ignore Git.
Sebelum masuk corpus, anonimisasi nama, nomor, lokasi spesifik, username, dan detail pribadi lain.
Template batch manual tersedia di `data/templates/manual_batch_template.txt`. Copy ke `data/private/manual_batch_001.txt`, isi dengan dialog buatan/kurasi sendiri, lalu masukkan saat build:

```bash
python tools/build_dataset.py \
  --max-rows-per-source 1000 \
  --synthesize 500 \
  --extra-text data/private/manual_batch_001.txt
```

Target batch manual yang disarankan:

- 40% curhat/emotional support,
- 20% obrolan santai/gabut,
- 15% tanya balik dan follow-up konteks,
- 10% roleplay ringan,
- 10% batas aman/menolak hal berbahaya secara halus,
- 5% dialek/slang lokal jika memang ingin style pack.

Untuk validation set, pilih contoh yang mirip kasus nyata tetapi **jangan** duplikat dari train. Valid set adalah ujian, bukan bahan belajar.

Audit corpus setiap selesai build:

```bash
python tools/audit_dataset.py
```

Output penting:

- `train_valid_overlap` harus 0,
- fail-level issue harus 0,
- valid idealnya 5-10% dari total dan minimal 500 dialog untuk training lebih serius,
- `synthetic_ratio` idealnya <= 30%,
- `response_template_repetition` jangan tinggi; kalau tinggi, model cenderung menjawab dengan frasa yang sama.

## Catatan sosial media

Jangan langsung scrape/copy TikTok, X/Twitter, Facebook, Instagram, atau forum tertutup untuk training kecuali kamu punya izin/lisensi yang jelas. Postingan publik tetap bisa punya hak cipta, data pribadi, dan batasan Terms of Service.

Jalur aman yang dipakai proyek ini:

1. gunakan dataset publik dengan lisensi jelas,
2. simpan raw sosial media di `data/raw/` jika perlu inspeksi lokal,
3. filter URL, handle, nama orang, politik, hinaan, PII, dan spam,
4. ubah hanya menjadi seed gaya/intent kecil, bukan mayoritas corpus,
5. audit + spot-check manual sebelum training.

Contoh memakai sumber social-style Apache-2.0 yang sudah ada di manifest tetapi disabled-by-default:

```bash
python tools/build_dataset.py \
  --sources nixia_seed,lorthgyu_indonesian_chat,w11wo_twitter_indonesia_sarcastic \
  --max-rows-per-source 1000 \
  --synthesize 500

python tools/audit_dataset.py
```

Kalau hasil audit menunjukkan template repetition tinggi atau banyak contoh politis/toxic, jangan fine-tune dulu; tambah curated manual yang lebih natural.

## Build kandidat long training

Command berikut menghasilkan corpus yang lulus audit awal untuk long training lokal di mesin development:

```bash
python tools/build_dataset.py \
  --sources nixia_seed,lorthgyu_indonesian_chat,lorthgyu_indonesian_qa,suryaadhi_ppmb_qa_id,w11wo_twitter_indonesia_sarcastic,gabrielb_python_qa,seacrowd_seadialogues \
  --allow-sharealike \
  --max-rows-per-source 6000 \
  --source-limit seacrowd_seadialogues=1200 \
  --synthesize 1500 \
  --valid-ratio 0.1 \
  --min-score 0.8 \
  --offline

python tools/audit_dataset.py
```

Hasil audit terakhir pada mesin ini:

```text
status=pass
readiness=longer_training_candidate
train=6258
valid=695
synthetic_ratio=21.4%
train_valid_overlap=0
```

Catatan penting:

- `--allow-sharealike` memasukkan `seacrowd_seadialogues` berlisensi CC-BY-SA; simpan `build_report.json` untuk atribusi dan pahami kewajiban ShareAlike jika dataset/model didistribusikan.
- `gabrielb_python_qa` besar dan domain teknis; jika output Nixia jadi terlalu coding-assistant, kurangi dengan `--source-limit gabrielb_python_qa=1500` atau hapus sumber itu.
- `--source-limit SOURCE_ID=N` membatasi satu sumber tanpa menurunkan sumber lain, berguna untuk mencegah satu dataset mendominasi.

Jika butuh variasi lokal, gunakan style pack terpisah seperti `data/style_packs/local_flavor_sample.txt`, atau aktifkan generator lokal secara eksplisit:

```bash
python tools/build_dataset.py \
  --include-local-flavor \
  --max-rows-per-source 1000 \
  --synthesize 2000
```

Kamu juga bisa memasukkan style pack/corpus lokal langsung tanpa mengubah `dataset_sources.json`:

```bash
python tools/build_dataset.py \
  --max-rows-per-source 1000 \
  --synthesize 3000 \
  --extra-text data/style_packs/local_flavor_sample.txt
```

`--extra-text` bisa diulang dan menerima format corpus biasa:

```text
<user> halo, kamu lagi apa?
<char> aku lagi santai nih hehe. kamu sendiri gimana?
```

`data/style_packs/chatfix_manual_seed.txt` berisi dialog manual original untuk mengoreksi model yang terlalu sering menjawab teknis/coding. Pakai bersama corpus chat-fix, bukan sebagai mayoritas dataset long training.

## Build corpus chat-fix rendah synthetic

Gunakan ini setelah model long terlalu bias ke coding/helpdesk. Sumber teknis dikeluarkan, social/SEADialogues dinaikkan, dan synthetic dijaga <30%:

```bash
python tools/build_dataset.py \
  --sources nixia_seed,lorthgyu_indonesian_chat,lorthgyu_indonesian_qa,w11wo_twitter_indonesia_sarcastic,seacrowd_seadialogues \
  --allow-sharealike \
  --max-rows-per-source 3000 \
  --source-limit seacrowd_seadialogues=1200 \
  --source-limit w11wo_twitter_indonesia_sarcastic=2000 \
  --synthesize 800 \
  --valid-ratio 0.1 \
  --min-score 0.8 \
  --offline \
  --extra-text data/style_packs/chatfix_manual_seed.txt \
  --output data/curated/chatfix_train.txt \
  --valid-output data/curated/chatfix_valid.txt \
  --report data/curated/chatfix_report.json

python tools/audit_dataset.py \
  --train data/curated/chatfix_train.txt \
  --valid data/curated/chatfix_valid.txt \
  --build-report data/curated/chatfix_report.json \
  --json-output data/curated/chatfix_audit.json
```

Hasil audit terakhir:

```text
readiness=small_finetune_candidate
train=2835
valid=314
synthetic_ratio=25.3%
train_valid_overlap=0
```

Warn ukuran train/valid masih wajar untuk fine-tune pendek, bukan long training. Pakai 1-2 epoch dan LR kecil.

Cendol bisa dipanggil eksplisit untuk pretraining/instruction:

```bash
python tools/build_dataset.py \
  --sources nixia_seed,lorthgyu_indonesian_chat,indonlp_cendol_chat_v2 \
  --max-rows-per-source 5000 \
  --synthesize 1000
```

## Build dengan sumber ShareAlike

```bash
python tools/build_dataset.py \
  --allow-sharealike \
  --max-rows-per-source 1000 \
  --synthesize 2000 \
  --output data/curated/train_corpus.txt \
  --valid-output data/curated/valid_corpus.txt
```

Jika memakai `--allow-sharealike`, distribusi dataset turunan mungkin wajib atribusi dan ShareAlike.

## Build besar

Untuk puluhan/ratusan ribu dialog:

```bash
python tools/build_dataset.py \
  --allow-sharealike \
  --max-rows-per-source 50000 \
  --synthesize 20000 \
  --target-dialogues 100000
```

Catatan: script stdlib ini aman dan ringan, tapi row API HF lambat untuk 100k+. Untuk produksi, lebih baik pakai `datasets` + parquet streaming, lalu gunakan filter yang sama.

## Training dari corpus hasil kurasi

```bash
cargo run --release -- tokenizer \
  --corpus data/curated/train_corpus.txt \
  --vocab artifacts/vocab.txt \
  --vocab-size 6000

cargo run --release -- train \
  --preset nixia-micro \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/nixia-micro \
  --epochs 10 \
  --batch-size 16
```

Training saat ini memakai Burn Flex CPU untuk checkpoint stabil. Jangan mengandalkan
`--features wgpu-backend` untuk training utama sampai jalur WGPU diverifikasi lagi.

## Prompt regression eval

Gunakan prompt tetap untuk membandingkan output sebelum/sesudah fine-tune:

```bash
python tools/eval_prompts.py \
  --artifacts artifacts/nixia-micro \
  --vocab artifacts/vocab.txt \
  --output data/curated/prompt_eval.md
```

Prompt ada di `data/eval_prompts.txt`. Tool ini bukan penilai otomatis; ia hanya menyimpan output dan flag kasar seperti prompt echo, repetisi, atau respons safety yang perlu review.

## Lanjut training

Ada dua mode lanjutan:

1. **Resume run yang sama**: pakai checkpoint di artifact yang sama, termasuk optimizer state.

```bash
cargo run --release -- train \
  --preset nixia-micro \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/nixia-micro \
  --resume-epoch 10 \
  --epochs 15 \
  --batch-size 16
```

2. **Fine-tune dari bobot lama**: pakai artifact baru dan optimizer baru.

```bash
cargo run --release -- train \
  --preset nixia-micro \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/nixia-micro-style \
  --init-from artifacts/nixia-micro \
  --epochs 2 \
  --batch-size 16 \
  --lr 0.00001
```

Gunakan train dari nol jika tokenizer/vocab berubah, preset/arsitektur berubah, atau dataset dasar dirombak besar.
Gunakan fine-tune jika hanya menambah data kecil/style pack dan `model_config.json` tetap kompatibel.

## Filter kebersihan

Builder menolak row yang:

- license tidak sesuai policy flag,
- mengandung URL/email/nomor telepon,
- terlalu panjang/pendek,
- terlalu repetitif,
- terlihat seperti code/table/markdown berat,
- berisi konten eksplisit/berbahaya dasar,
- terlalu jauh dari bahasa Indonesia/slang/lokal.

Output tetap perlu spot-check manual sebelum training serius.
