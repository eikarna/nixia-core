use std::path::Path;
use turbovec::IdMapIndex;

/// RAG Database implementation using TurboVec for fast vector search.
pub struct RagDatabase {
    index: IdMapIndex,
    dimension: usize,
    #[allow(dead_code)]
    bits: usize,
}

impl RagDatabase {
    /// Creates a new RAG database with the specified embedding dimension and quantization bits (2 or 4).
    pub fn new(dimension: usize, bits: usize) -> Self {
        Self {
            index: IdMapIndex::new(dimension, bits).expect("Failed to create IdMapIndex"),
            dimension,
            bits,
        }
    }

    /// Loads a previously saved RAG database from disk.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let index = IdMapIndex::load(path.as_ref()).map_err(|e| e.to_string())?;
        // We assume 2 or 4 bits depending on how it was saved, but dimension is internal.
        Ok(Self {
            index,
            dimension: 0, // In a real scenario we'd query the index or store metadata.
            bits: 0,
        })
    }

    /// Saves the RAG database to disk.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        self.index.write(path.as_ref()).map_err(|e| e.to_string())
    }

    /// Adds new documents (embeddings) with their corresponding IDs to the index.
    pub fn add_documents(&mut self, embeddings: &[f32], ids: &[u64]) {
        assert_eq!(
            embeddings.len(),
            ids.len() * self.dimension,
            "Embeddings size mismatch"
        );
        self.index
            .add_with_ids(embeddings, ids)
            .expect("Failed to add vectors");
    }

    /// Searches for the top_k most similar documents to the given query embeddings.
    /// `queries` should be a flattened slice of f32 vectors.
    pub fn search(&self, queries: &[f32], top_k: usize) -> (Vec<f32>, Vec<u64>) {
        self.index.search(queries, top_k)
    }

    /// Removes a document by its ID.
    pub fn remove_document(&mut self, id: u64) {
        self.index.remove(id);
    }
}
