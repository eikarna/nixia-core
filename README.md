# Nixia

Nixia adalah proyek awal tiny causal language model Bahasa Indonesia menggunakan Rust dan Burn.
Target desainnya adalah model kecil untuk eksperimen on-device, terutama perangkat Android low-end seperti Xiaomi Redmi 4X.

## Isi proyek

- `src/tokenizer`: normalizer, greedy subword tokenizer, dan BPE trainer sederhana.
- `src/data`: pembaca corpus dan dataset language modeling.
- `src/model`: decoder-only Tiny Transformer dengan RMSNorm + SwiGLU.
- `src/training`: konfigurasi, loop training Burn, dan evaluasi loss/perplexity.
- `src/inference`: loading model, chat template, sampling, dan weight-only int8 quantization helper.

## Backend

Training sengaja memakai Burn Flex CPU untuk checkpoint yang stabil dan portabel. Feature
`wgpu-backend` tetap bisa dikompilasi untuk eksperimen backend, tetapi command `train`
tidak memakai WGPU karena sebelumnya checkpoint hasil WGPU mudah menjadi NaN di proyek ini.

Gunakan command training tanpa feature WGPU:

```bash
cargo run --release -- train --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run
```

## Quick start

```bash
cargo run -- tokenizer --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --vocab-size 8000
cargo run -- train --preset dev-smoke --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run --epochs 1 --batch-size 2
cargo run -- eval --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run
cargo run -- generate --chat --artifacts artifacts/run --vocab artifacts/vocab.txt --prompt "halo, kamu siapa?" --tokens 40
```

Sample corpus hanya untuk smoke test, bukan untuk menghasilkan model bagus.

Prompt regression eval untuk membandingkan output antar run:

```bash
python tools/eval_prompts.py --artifacts artifacts/run --vocab artifacts/vocab.txt
```

Prompt tetap ada di `data/eval_prompts.txt`; output default ditulis ke `data/curated/prompt_eval.md`.

## Profil model yang disarankan

Untuk target low-end, mulai dari profil kecil dahulu:

```bash
--preset nixia-micro
```

Detail preset:

```text
vocab_size: sesuai tokenizer, disarankan 6000
seq_len: 96
d_model: 192
layers: 6
heads: 4
d_ff: 512
```

Jika performa masih cukup:

```bash
--preset nixia-tiny
```

Detail preset:

```text
vocab_size: sesuai tokenizer, disarankan 8000
seq_len: 128
d_model: 256
layers: 8
heads: 4
d_ff: 768
```

Preset `dev-smoke` hanya untuk validasi pipeline cepat.

## Format corpus

Gunakan format dialog eksplisit:

```text
<user> halo, kamu lagi apa?
<char> aku lagi santai nih hehe. kamu sendiri gimana?
```

Untuk roleplay, campur narasi pendek:

```text
<char> *tersenyum kecil* aku ngerti kok, kamu capek ya?
```

## Dataset builder

Untuk menyusun corpus chat yang lebih matang dari sumber yang sudah dicek:

```bash
python tools/build_dataset.py --max-rows-per-source 1000 --synthesize 5000
```

Output default:

- `data/curated/train_corpus.txt`
- `data/curated/valid_corpus.txt`
- `data/curated/build_report.json`

File `.txt` di proyek ini hanya file data/plain-text biasa. Training tidak otomatis membaca semua `.txt`;
yang dipakai hanya file yang kamu berikan lewat `--corpus`, `--valid`, atau `--extra-text` saat build dataset.
Overfitting muncul karena data terlalu sedikit, repetitif, atau epoch terlalu panjang, bukan karena jumlah file `.txt` di folder.

Audit kualitas sebelum training panjang:

```bash
python tools/audit_dataset.py
```

Audit mengecek format `<user>/<char>`, duplikat, train/valid overlap, ukuran valid set,
rasio synthetic, dan repetisi template respons.

Sumber dataset dan catatan lisensi ada di `data/DATASET_GUIDE.md` dan `data/dataset_sources.json`.
Gunakan `--allow-sharealike` hanya jika kamu siap mengikuti kewajiban atribusi/ShareAlike.
Base corpus sengaja netral-kasual; slang/dialek berat sebaiknya masuk style pack atau fine-tuning persona terpisah supaya persona tidak bocor dan tidak context rot.

Tambahkan corpus/style pack lokal tanpa mengubah manifest dengan `--extra-text`:

```bash
python tools/build_dataset.py --max-rows-per-source 1000 --synthesize 3000 --extra-text data/style_packs/local_flavor_sample.txt
```

Corpus koreksi gaya chat manual tersedia di `data/style_packs/chatfix_manual_seed.txt`. Ini original/project-local dan bisa dipakai untuk fine-tune pendek saat model terlalu sering menjawab seperti coding/helpdesk assistant.

Untuk data buatan/kurasi sendiri, copy `data/templates/manual_batch_template.txt` ke
`data/private/manual_batch_001.txt`, isi dan anonimisasi, lalu build dengan `--extra-text`.
Folder `data/private/` dan `data/raw/` sengaja tidak masuk Git.

Untuk gaya sosial media, jangan scrape TikTok/X/Facebook langsung kecuali kamu punya izin/lisensi yang jelas.
Gunakan sumber publik yang lisensinya kompatibel dan tetap disabled-by-default, misalnya:

```bash
python tools/build_dataset.py --sources nixia_seed,lorthgyu_indonesian_chat,w11wo_twitter_indonesia_sarcastic --max-rows-per-source 1000 --synthesize 500
```

Setelah itu wajib jalankan `python tools/audit_dataset.py` dan spot-check manual.

Dataset kandidat untuk long training lokal bisa dibuat dengan mix lebih besar berikut:

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

Catatan: command ini memasukkan sumber CC-BY-SA, jadi distribusi dataset/model turunan mungkin punya kewajiban atribusi/ShareAlike. Untuk penggunaan privat lokal, tetap simpan report lisensi.

Untuk chat-clean/chat-fix, gunakan mix tanpa sumber teknis, synthetic rendah, dan tokenizer baru jika train dari nol:

```bash
python tools/build_dataset.py \
  --sources nixia_seed,lorthgyu_indonesian_chat,lorthgyu_indonesian_qa,w11wo_twitter_indonesia_sarcastic,seacrowd_seadialogues \
  --allow-sharealike \
  --max-rows-per-source 3000 \
  --source-limit seacrowd_seadialogues=150 \
  --source-limit w11wo_twitter_indonesia_sarcastic=2000 \
  --synthesize 800 \
  --synth-mode chat-clean \
  --valid-ratio 0.1 \
  --min-score 0.8 \
  --offline \
  --extra-text data/style_packs/chatfix_manual_seed.txt \
  --output data/curated/chatclean_train.txt \
  --valid-output data/curated/chatclean_valid.txt \
  --report data/curated/chatclean_report.json

python tools/audit_dataset.py \
  --train data/curated/chatclean_train.txt \
  --valid data/curated/chatclean_valid.txt \
  --build-report data/curated/chatclean_report.json \
  --json-output data/curated/chatclean_audit.json
```

Hasil audit terakhir untuk chat-clean:

```text
readiness=small_finetune_candidate
train=1582
valid=175
synthetic_ratio=25.5%
train_valid_overlap=0
```

Fine-tune pendek dari model long:

```bash
cargo run --release -- train \
  --preset nixia-micro \
  --corpus data/curated/chatclean_train.txt \
  --valid data/curated/chatclean_valid.txt \
  --vocab artifacts/vocab-long.txt \
  --artifacts artifacts/nixia-micro-chatfix \
  --init-from artifacts/nixia-micro-long \
  --epochs 2 \
  --batch-size 16 \
  --lr 0.00001
```

Untuk model casual/chat dari nol, lebih baik train dari corpus chat-clean dan tokenizer baru:

```bash
cargo run --release -- tokenizer \
  --corpus data/curated/chatclean_train.txt \
  --vocab artifacts/vocab-chatclean.txt \
  --vocab-size 4000

cargo run --release -- train \
  --preset nixia-micro \
  --corpus data/curated/chatclean_train.txt \
  --valid data/curated/chatclean_valid.txt \
  --vocab artifacts/vocab-chatclean.txt \
  --artifacts artifacts/nixia-micro-chatclean \
  --epochs 15 \
  --batch-size 16 \
  --lr 0.00005
```

## Long training end-to-end

Command di bawah memakai format Windows `cmd` dengan `^` sebagai line continuation. Jika memakai Git Bash/Linux/macOS, ganti `^` dengan `\`.

1. Build corpus kandidat long training:

```bat
python tools/build_dataset.py ^
  --sources nixia_seed,lorthgyu_indonesian_chat,lorthgyu_indonesian_qa,suryaadhi_ppmb_qa_id,w11wo_twitter_indonesia_sarcastic,gabrielb_python_qa,seacrowd_seadialogues ^
  --allow-sharealike ^
  --max-rows-per-source 6000 ^
  --source-limit seacrowd_seadialogues=1200 ^
  --synthesize 1500 ^
  --valid-ratio 0.1 ^
  --min-score 0.8 ^
  --offline
```

2. Audit dataset. Lanjut training hanya jika `status=pass` dan `readiness=longer_training_candidate`:

```bat
python tools/audit_dataset.py
```

3. Buat tokenizer baru untuk dataset ini. Pakai file vocab baru agar artifact lama tidak tertukar:

```bat
cargo run --release -- tokenizer ^
  --corpus data/curated/train_corpus.txt ^
  --vocab artifacts/vocab-long.txt ^
  --vocab-size 6000
```

4. Train dari nol dengan preset `nixia-micro`:

```bat
cargo run --release -- train ^
  --preset nixia-micro ^
  --corpus data/curated/train_corpus.txt ^
  --valid data/curated/valid_corpus.txt ^
  --vocab artifacts/vocab-long.txt ^
  --artifacts artifacts/nixia-micro-long ^
  --epochs 15 ^
  --batch-size 16 ^
  --lr 0.00005
```

Jika RAM/waktu tidak cukup, turunkan `--batch-size 8`. Jika training terhenti setelah checkpoint epoch tertentu, lanjutkan seperti ini:

```bat
cargo run --release -- train ^
  --preset nixia-micro ^
  --corpus data/curated/train_corpus.txt ^
  --valid data/curated/valid_corpus.txt ^
  --vocab artifacts/vocab-long.txt ^
  --artifacts artifacts/nixia-micro-long ^
  --resume-epoch 10 ^
  --epochs 15 ^
  --batch-size 16 ^
  --lr 0.00005
```

5. Evaluasi validation loss/perplexity:

```bat
cargo run --release -- eval ^
  --corpus data/curated/valid_corpus.txt ^
  --vocab artifacts/vocab-long.txt ^
  --artifacts artifacts/nixia-micro-long
```

Opsional, cek train loss untuk melihat jarak train-vs-valid:

```bat
cargo run --release -- eval ^
  --corpus data/curated/train_corpus.txt ^
  --vocab artifacts/vocab-long.txt ^
  --artifacts artifacts/nixia-micro-long
```

6. Jalankan prompt regression eval:

```bat
python tools/eval_prompts.py ^
  --artifacts artifacts/nixia-micro-long ^
  --vocab artifacts/vocab-long.txt ^
  --output data/curated/prompt_eval_long.md
```

7. Tes generate/chat manual:

```bat
cargo run --release -- generate ^
  --chat ^
  --artifacts artifacts/nixia-micro-long ^
  --vocab artifacts/vocab-long.txt ^
  --prompt "aku capek banget hari ini, rasanya pengen tidur tapi pikiran rame" ^
  --tokens 64 ^
  --temperature 0.8 ^
  --top-k 30 ^
  --top-p 0.92 ^
  --min-p 0.03
```

Contoh prompt lain:

```bat
cargo run --release -- generate --chat --artifacts artifacts/nixia-micro-long --vocab artifacts/vocab-long.txt --prompt "halo, kamu siapa?" --tokens 64
cargo run --release -- generate --chat --artifacts artifacts/nixia-micro-long --vocab artifacts/vocab-long.txt --prompt "temenin aku diem dulu boleh?" --tokens 64
```

Tanda training layak diteruskan: valid loss turun/stabil, prompt eval makin natural, dan output tidak makin repetitif. Stop atau turunkan LR kalau train loss turun tetapi valid loss naik.

## Lanjut training, fine-tune, atau dari nol

- Pakai `--resume-epoch N` untuk melanjutkan run yang sama dari checkpoint `artifacts/run/checkpoint/*-N.mpk`. Ini memuat model, optimizer, dan scheduler state.
- Pakai `--init-from artifacts/base` untuk fine-tune bobot model lama ke artifact baru. Ini memuat bobot model saja dan membuat optimizer baru.
- Train dari nol jika tokenizer/vocab berubah, preset/arsitektur berubah, atau dataset dirombak besar-besaran.
- Fine-tune cocok jika hanya menambah data kecil/style pack dan vocab/model config tetap sama.

Contoh fine-tune aman ke artifact baru:

```bash
cargo run --release -- train \
  --preset nixia-micro \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/nixia-micro-v2 \
  --init-from artifacts/nixia-micro \
  --epochs 2 \
  --batch-size 16 \
  --lr 0.00001
```

Contoh true resume dari epoch 10 ke 15:

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

## Catatan Android

Integrasi JNI belum dibuat. Untuk Android low-end, rencana runtime terbaik adalah:

- Burn Flex CPU lebih dulu.
- Batch size 1.
- Context 64-128 token.
- Sampling maksimal 32-64 token per respons.
- Quantization weight-only int8 setelah akurasi model f32/f16 stabil.

## Sampling

Generator mendukung sampling yang lebih aman untuk model kecil:

- `--temperature`, default `0.8`
- `--top-k`, default `30`
- `--top-p`, default `0.92`
- `--min-p`, default `0.03`
- repetition penalty dan no-repeat trigram aktif di kode default

Gunakan `--chat` agar prompt otomatis dibungkus menjadi:

```text
<user> pesan kamu <char>
```
