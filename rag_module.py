import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from config import model # Memanggil model dari config.py

def retrieve_similar_case(current_entities_json, rag_kb):
    print("Mencari referensi kasus serupa di Vector Database (Knowledge Base)...")
    
    target_text = str(current_entities_json)
    kb_texts = [doc["page_content"] for doc in rag_kb]
    
    try:
        # 3. Buat Vector Embedding menggunakan model khusus embedding Gemini
        print("Membuat vektor untuk Knowledge Base...")
        kb_embeddings = genai.embed_content(
            model="models/gemini-embedding-001", # <-- UBAH DI SINI
            content=kb_texts,
            task_type="retrieval_document"
        )['embedding']
        
        target_embedding = genai.embed_content(
            model="models/gemini-embedding-001", # <-- UBAH JUGA DI SINI
            content=target_text,
            task_type="retrieval_query"
        )['embedding']
        
        kemiripan = cosine_similarity([target_embedding], kb_embeddings)[0]
        index_terbaik = np.argmax(kemiripan)
        dokumen_terbaik = rag_kb[index_terbaik]
        
        print(f"Kasus serupa ditemukan! (Kemiripan: {kemiripan[index_terbaik]:.2f})")
        return dokumen_terbaik["page_content"]
        
    except Exception as e:
        print(f"Error saat Retrieval (Pencarian): {e}")
        return "Tidak ada referensi kasus serupa yang ditemukan."

def generate_client_summary_rag(entities_json, recommendation_text, retrieved_context):
    print("Memulai tugas Generasi Ringkasan (RAG) dengan Gemini...")
    
    generation_config = GenerationConfig(temperature=0.2, top_p=0.8, top_k=40)

    prompt_template = f"""
    Anda adalah seorang asisten psikolog yang suportif.
    Tugas Anda menulis ringkasan laporan psikologis yang mudah dipahami klien.

    Gunakan 3 informasi berikut untuk menyusun jawaban:

    1. TEMUAN KLIEN SAAT INI (Hasil NER):
    {entities_json}

    2. RENCANA INTERVENSI DARI PSIKOLOG (Teks Asli):
    {recommendation_text}

    3. REFERENSI KASUS SERUPA (Hasil Pencarian RAG dari Knowledge Base):
    Gunakan referensi ini sebagai inspirasi tambahan jika relevan dengan kondisi klien saat ini:
    {retrieved_context}

    Instruksi:
    - Jelaskan temuan utama secara suportif.
    - Sebutkan kekuatan klien.
    - Berikan rekomendasi langkah-langkah berdasarkan 'Rencana Intervensi' dan diperkaya dengan 'Referensi Kasus Serupa'.
    - JANGAN mengulang teks mentah, buatlah ringkasan naratif yang natural.

    Ringkasan untuk Klien:
    """
    try:
        response = model.generate_content(prompt_template, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"Error API Gemini saat Generasi: {e}")
        return None