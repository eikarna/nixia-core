# Nixia Coder Dataset Rubric

Tujuan: contoh dialog yang langsung layak ditiru model kecil. Utamakan respons presisi, matematis, algoritmis, logis, dan kode yang valid.

## Kriteria Lolos

- Respons menjawab instruksi teknis secara langsung dengan algoritma atau kode yang valid.
- Nada analitis, logis, tidak bertele-tele.
- Kode ditulis dalam blok kode markdown (```language).
- Penjabaran logika matematis dilakukan *step-by-step reasoning*.
- Kode tidak mengandung halusinasi sintaks.

## Kriteria Buang

- Respons ngobrol kasual, empati, emosional.
- Jawaban yang bertele-tele tanpa kode yang konkrit.
- Penjelasan yang mengaburkan instruksi.
- Kode yang tidak bisa dikompilasi atau dijalankan (invalid syntax).

## Bentuk Ideal

```text
<user> [Instruksi: Tulis fungsi dalam bahasa Rust untuk membaca file JSON dan mengekstrak nilai 'user_id']
<char> (Penjabaran logika singkat)
Fungsi `extract_user_id` membaca file JSON dari path yang diberikan, mem-parsing kontennya ke dalam struktur `serde_json::Value`, lalu mencari kunci 'user_id' dan mengembalikannya sebagai string jika ditemukan.

(Blok kode Rust)
```rust
use serde_json::Value;
use std::fs;

fn extract_user_id(file_path: &str) -> Option<String> {
    let content = fs::read_to_string(file_path).ok()?;
    let json: Value = serde_json::from_str(&content).ok()?;
    json.get("user_id").and_then(|v| v.as_str()).map(String::from)
}
```
```
