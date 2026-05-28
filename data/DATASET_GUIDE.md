# Nixia Dataset Guide

Tujuan dataset Nixia adalah chat Bahasa Indonesia kasual/roleplay ringan yang tetap bersih, legal, dan cocok untuk model kecil Redmi 4X.

## Sumber yang sudah dicek

| ID | Sumber | License | Default | Catatan |
|---|---|---:|---:|---|
| `nixia_seed` | `data/sample_corpus.txt` | project-local | ya | Seed manual gaya Nixia |
| `lorthgyu_indonesian_chat` | HF `LorthGyu/indonesian-chat` | MIT | ya | 200 multi-turn chat Indo |
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
  --preset redmi-nano \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/redmi-nano \
  --epochs 10 \
  --batch-size 16
```

Training saat ini memakai Burn Flex CPU untuk checkpoint stabil. Jangan mengandalkan
`--features wgpu-backend` untuk training utama sampai jalur WGPU diverifikasi lagi.

## Lanjut training

Ada dua mode lanjutan:

1. **Resume run yang sama**: pakai checkpoint di artifact yang sama, termasuk optimizer state.

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

2. **Fine-tune dari bobot lama**: pakai artifact baru dan optimizer baru.

```bash
cargo run --release -- train \
  --preset redmi-nano \
  --corpus data/curated/train_corpus.txt \
  --valid data/curated/valid_corpus.txt \
  --vocab artifacts/vocab.txt \
  --artifacts artifacts/redmi-nano-style \
  --init-from artifacts/redmi-nano \
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
