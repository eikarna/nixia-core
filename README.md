# Nixia - Indonesian Logic, Math, and Programming Engine

Nixia adalah proyek awal tiny causal language model Bahasa Indonesia menggunakan Rust dan Burn. Model ini difokuskan sebagai "Logic, Math, and Programming Engine", ditujukan secara spesifik untuk penalaran algoritmis dan pemrograman.
Target desainnya adalah model kecil untuk eksperimen on-device, terutama perangkat Android low-end seperti Xiaomi Redmi 4X.

## Isi proyek

- `src/tokenizer`: normalizer, greedy subword tokenizer, dan BPE trainer sederhana.
- `src/data`: pembaca corpus dan dataset language modeling.
- `src/model`: decoder-only Tiny Transformer dengan RMSNorm + SwiGLU.
- `src/training`: konfigurasi, loop training Burn, dan evaluasi loss/perplexity.
- `src/inference`: loading model, chat template, sampling, dan weight-only int8 quantization helper.

## Backend dan resource training

Training default memakai Burn Flex CPU untuk checkpoint yang stabil dan portabel. Untuk
eksperimen GPU, command `train` mendukung pilihan backend:

- `--backend flex` default CPU/Flex.
- `--backend wgpu` untuk AMD Radeon, Intel GPU, dan GPU lain yang didukung WGPU.
- `--backend cuda` untuk NVIDIA/CUDA.
- `--backend rocm` untuk AMD ROCm di Linux yang kompatibel.

Backend GPU harus dikompilasi dengan feature Cargo yang sesuai:

```bash
cargo run --release --features wgpu-backend -- train --backend wgpu --gpu-kind discrete --device-index 0 ...
cargo run --release --features cuda-backend -- train --backend cuda --device-index 0 ...
cargo run --release --features rocm-backend -- train --backend rocm --device-index 0 ...
```

Resource training dikontrol lewat `--batch-size`, `--seq-len`, ukuran model, backend,
`--device-index`, `--gpu-kind`, dan `--num-workers`. Untuk memaksimalkan pemakaian GPU,
naikkan `--batch-size` bertahap sampai VRAM hampir penuh tetapi training masih stabil.
Tidak ada jaminan utilisasi 99.9% karena scheduler GPU/driver dan ukuran model ikut menentukan.

Gunakan command training CPU tanpa feature GPU:

```bash
cargo run --release -- train --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run
```

Notebook Kaggle/Colab tersedia di `notebooks/nixia_train_gpu.ipynb`.

## Quick start

```bash
cargo run -- tokenizer --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --vocab-size 8000
cargo run -- train --preset dev-smoke --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run --epochs 1 --batch-size 2
cargo run -- eval --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run
cargo run -- generate --chat --artifacts artifacts/run --vocab artifacts/vocab.txt --prompt "bagaimana cara mendeklarasikan variabel mutable di rust?" --tokens 40
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
yang dipakai hanya file yang kamu berikan lewat `--corpus`, `--valid`, `--extra-text`, atau `--extra-glob` saat build dataset.
Overfitting muncul karena data terlalu sedikit, repetitif, atau epoch terlalu panjang, bukan karena jumlah file `.txt` di folder.

## Mengelola dataset sendiri

Dataset manual/project-local bisa disimpan di `data/templates/nixia_dataset_*.txt`. Saat ini ada 10 batch (`001`-`010`), masing-masing sekitar 100 dialog. File ini aman untuk version control jika isinya original dan bukan chat pribadi. Untuk chat pribadi/raw, simpan di `data/private/` karena folder itu di-ignore Git.

Format wajib tiap dialog:

```text
<user> pesan user
<char> balasan nixia

<user> dialog berikutnya
<char> balasan berikutnya
```

Melihat ringkasan/filter dataset manual tanpa menulis file:

```bash
python tools/build_dataset.py \
  --sources sahil2801_codealpaca,gabrielb_python_qa \
  --max-rows-per-source 0 \
  --synthesize 0 \
  --min-score 0.8 \
  --offline \
  --extra-glob "data/templates/nixia_dataset_*.txt" \
  --dry-run
```

Menambahkan batch baru:

1. Copy `data/templates/manual_batch_template.txt` menjadi `data/templates/nixia_dataset_011.txt`.
2. Isi 50-100 dialog original.
3. Jalankan dry-run di atas dan cek `accepted`/`reject_*`.

Mengurangi dataset:

- Hapus atau pindahkan file batch dari pola `data/templates/nixia_dataset_*.txt`.
- Atau rename sementara, misalnya `nixia_dataset_011.disabled.txt`, supaya tidak ikut `--extra-glob`.

Merevisi dataset:

- Edit file batch langsung.
- Jaga role `<user>`/`<char>`, pisahkan dialog dengan baris kosong, dan hindari PII/link/konten berbahaya.
- Jalankan build + audit ulang sebelum training.

Build corpus bersih dari semua dataset manual. Command ini sengaja tidak mengambil sumber publik/social supaya vocab tidak ikut menyerap nama orang, simbol mojibake, atau gaya forum yang noisy:

```bash
python tools/build_dataset.py \
  --sources sahil2801_codealpaca,gabrielb_python_qa \
  --max-rows-per-source 0 \
  --synthesize 800 \

  --valid-ratio 0.1 \
  --min-score 0.8 \
  --offline \
  --extra-text data/templates/nixia_coder_001.txt \
  --extra-glob "data/templates/nixia_dataset_*.txt" \
  --output data/curated/chatclean_train.txt \
  --valid-output data/curated/chatclean_valid.txt \
  --report data/curated/chatclean_report.json

python tools/audit_dataset.py \
  --train data/curated/chatclean_train.txt \
  --valid data/curated/chatclean_valid.txt \
  --build-report data/curated/chatclean_report.json \
  --json-output data/curated/chatclean_audit.json
```

Audit kualitas sebelum training panjang:

```bash
python tools/audit_dataset.py
```

Audit mengecek format `<user>/<char>`, duplikat, train/valid overlap, ukuran valid set,
rasio synthetic, dan repetisi template respons.

Sumber dataset dan catatan lisensi ada di `data/DATASET_GUIDE.md` dan `data/dataset_sources.json`.
Gunakan `--allow-sharealike` hanya jika kamu siap mengikuti kewajiban atribusi/ShareAlike.
Base corpus difokuskan pada penalaran logis, matematika, dan pemrograman. Data kasual atau slang sangat dibatasi agar kemampuan algoritmis model tidak terganggu.

Tambahkan corpus/style pack lokal tanpa mengubah manifest dengan `--extra-text`:

```bash
python tools/build_dataset.py --max-rows-per-source 1000 --synthesize 3000 --extra-text data/style_packs/local_flavor_sample.txt
```

Corpus referensi coder tersedia di `data/templates/nixia_coder_001.txt`. Ini original/project-local dan bisa dipakai sebagai contoh standar untuk fine-tune pendek agar konsisten dalam menjawab instruksi logika/koding.

Untuk data buatan/kurasi sendiri, copy `data/templates/manual_batch_template.txt` ke
`data/private/manual_batch_001.txt`, isi dan anonimisasi, lalu build dengan `--extra-text`.
Folder `data/private/` dan `data/raw/` sengaja tidak masuk Git.

Untuk data teknis, utamakan dataset instruksional dan dokumentasi teknis dengan lisensi yang jelas.
Gunakan sumber publik yang lisensinya kompatibel dan tetap disabled-by-default, misalnya:

```bash
python tools/build_dataset.py --sources sahil2801_codealpaca,gabrielb_python_qa --max-rows-per-source 5000 --synthesize 0
```

Setelah itu wajib jalankan `python tools/audit_dataset.py` dan spot-check manual.

Dataset kandidat untuk long training lokal bisa dibuat dengan mix lebih besar berikut:

```bash
python tools/build_dataset.py \
  --sources sahil2801_codealpaca,gabrielb_python_qa \
  --allow-sharealike \
  --max-rows-per-source 6000 \

  --synthesize 1500 \
  --valid-ratio 0.1 \
  --min-score 0.8 \
  --offline

python tools/audit_dataset.py
```

Catatan: command ini memasukkan sumber CC-BY-SA, jadi distribusi dataset/model turunan mungkin punya kewajiban atribusi/ShareAlike. Untuk penggunaan privat lokal, tetap simpan report lisensi.

Untuk training coder dari nol, gunakan corpus spesifik koding dan logika dengan tingkat presisi tinggi. Hindari mencampur dataset kasual atau sosial:

```bash
python tools/build_dataset.py \
  --sources sahil2801_codealpaca,gabrielb_python_qa \
  --max-rows-per-source 0 \
  --synthesize 800 \

  --valid-ratio 0.1 \
  --min-score 0.8 \
  --offline \
  --extra-text data/templates/nixia_coder_001.txt \
  --extra-glob "data/templates/nixia_dataset_*.txt" \
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
train=1413
valid=157
synthetic_ratio=28.5%
train_valid_overlap=0
```

Catatan vocab: token seperti `▁aku`, `anyaan`, atau `eekend` adalah subword BPE normal; `▁` berarti ada spasi sebelum token. Yang perlu dihindari adalah mojibake/simbol asing dan nama orang dari corpus noisy. Builder sekarang menormalisasi `%`, `+`, `&`, escape aneh, mojibake, dan nama setelah sapaan seperti `Uda Reza` -> `Uda`.

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

Untuk model logika/coding dari nol, lebih baik train dari corpus spesifik coder dan tokenizer baru:

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
  --prompt "tuliskan fungsi python untuk menghitung bilangan prima dari 1 hingga n" ^
  --tokens 64 ^
  --temperature 0.8 ^
  --top-k 30 ^
  --top-p 0.92 ^
  --min-p 0.03
```

Contoh prompt lain:

```bat
cargo run --release -- generate --chat --artifacts artifacts/nixia-micro-long --vocab artifacts/vocab-long.txt --prompt "bagaimana cara mendeklarasikan variabel mutable di rust?" --tokens 64
cargo run --release -- generate --chat --artifacts artifacts/nixia-micro-long --vocab artifacts/vocab-long.txt --prompt "apa perbedaan antara TCP dan UDP?" --tokens 64
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
