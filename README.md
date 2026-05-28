# Nixia

Nixia adalah proyek awal tiny causal language model Bahasa Indonesia menggunakan Rust dan Burn.
Target desainnya adalah model kecil untuk eksperimen on-device, terutama perangkat Android low-end.

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

Untuk Redmi 4X, mulai dari profil kecil dahulu:

```bash
--preset redmi-nano
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
--preset redmi-tiny
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

## Lanjut training, fine-tune, atau dari nol

- Pakai `--resume-epoch N` untuk melanjutkan run yang sama dari checkpoint `artifacts/run/checkpoint/*-N.mpk`. Ini memuat model, optimizer, dan scheduler state.
- Pakai `--init-from artifacts/base` untuk fine-tune bobot model lama ke artifact baru. Ini memuat bobot model saja dan membuat optimizer baru.
- Train dari nol jika tokenizer/vocab berubah, preset/arsitektur berubah, atau dataset dirombak besar-besaran.
- Fine-tune cocok jika hanya menambah data kecil/style pack dan vocab/model config tetap sama.

Contoh fine-tune aman ke artifact baru:

```bash
cargo run --release -- train \
  --preset redmi-nano \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/redmi-nano-v2 \
  --init-from artifacts/redmi-nano \
  --epochs 2 \
  --batch-size 16 \
  --lr 0.00001
```

Contoh true resume dari epoch 10 ke 15:

```bash
cargo run --release -- train \
  --preset redmi-nano \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/redmi-nano \
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
