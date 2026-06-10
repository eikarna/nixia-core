use nixia::rag::RagDatabase;

fn main() {
    println!("Nixia Coder 1B");

    // Example TurboVec integration check
    let mut rag_db = RagDatabase::new(1536, 4);
    let sample_embed = vec![0.0f32; 1536];
    rag_db.add_documents(&sample_embed, &[42]);
    let (_, ids) = rag_db.search(&sample_embed, 1);
    println!("RAG Retrieved ID: {:?}", ids);
}
