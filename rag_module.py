import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from config import model # Memanggil model dari config.py
from google.api_core.exceptions import ResourceExhausted

def retrieve_similar_case(current_entities_json, rag_kb):
    print("Mengambil referensi dari Knowledge Base (Mode Stabil Tanpa API Embedding)...")
    
    try:
        # Menyatukan seluruh isi Knowledge Base untuk langsung diberikan ke model generatif
        # Langkah ini menghilangkan penggunaan library tambahan dan mengamankan kuota API
        semua_kasus = "\n\n---\n\n".join([doc["page_content"] for doc in rag_kb])
        print("✅ Referensi Knowledge Base berhasil dimuat secara lokal!")
        return semua_kasus
        
    except Exception as e:
        print(f"❌ Error saat memuat referensi: {e}")
        return "Tidak ada referensi kasus yang ditemukan."


def generate_client_summary_baseline(entities_json, recommendation_text):
    print("Memulai tugas Generasi Ringkasan (BASELINE - TANPA RAG)...")
    
    generation_config = GenerationConfig(temperature=0.2, top_p=0.8, top_k=40)

    prompt_template = f"""
    Anda adalah seorang asisten psikolog yang suportif.
    Tugas Anda menulis ringkasan laporan psikologis yang mudah dipahami klien.

    Gunakan 2 informasi berikut untuk menyusun jawaban:

    1. TEMUAN KLIEN SAAT INI (Hasil NER):
    {entities_json}

    2. RENCANA INTERVENSI DARI PSIKOLOG (Teks Asli):
    {recommendation_text}

    Instruksi:
    - Jelaskan temuan utama secara suportif.
    - Sebutkan kekuatan klien.
    - Berikan rekomendasi langkah-langkah secara eksklusif berdasarkan 'Rencana Intervensi' yang diberikan.
    - JANGAN mengulang teks mentah, buatlah ringkasan naratif yang natural.

    Ringkasan untuk Klien:
    """
    try:
        response = model.generate_content(prompt_template, generation_config=generation_config)
        return response.text
        
    except ResourceExhausted:
        print("❌ PERINGATAN: Terkena Limit Kuota API (429) saat Generasi Baseline!")
        return None
    except Exception as e:
        print(f"❌ Error API Gemini saat Generasi Baseline: {e}")
        return None


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

    3. REFERENSI KASUS SERUPA (Knowledge Base Lengkap):
    Gunakan kumpulan referensi berikut sebagai inspirasi tambahan HANYA JIKA relevan dengan kondisi klien saat ini:
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
        
    except ResourceExhausted:
        print("❌ PERINGATAN: Terkena Limit Kuota API (429) saat Generasi RAG!")
        return None
    except Exception as e:
        print(f"❌ Error API Gemini saat Generasi RAG: {e}")
        return None