import json
from config import model # Memanggil model dari config.py

def extract_entities_with_gemini(text_to_analyze, ner_kb):
    print("Memulai tugas NER dengan Gemini + Knowledge Base...")
    
    entity_definitions = """
    - "GEJALA": Tanda atau simptom klinis yang dilaporkan.
    - "DIAGNOSIS": Pola kepribadian atau sindrom klinis yang teridentifikasi.
    - "TANTANGAN": Faktor internal atau eksternal yang menghambat.
    - "KEKUATAN": Faktor positif atau area pemberdayaan.
    """

    prompt_template = f"""
    Anda adalah asisten psikologi yang bertugas melakukan Named Entity Recognition (NER).
    
    Definisi Entitas:
    {entity_definitions}

    KAMUS REFERENSI (Knowledge Base):
    Gunakan daftar istilah berikut sebagai panduan utama Anda untuk mengenali entitas:
    {json.dumps(ner_kb, indent=2)}

    Aturan Keluaran:
    - Ekstrak entitas dari teks, prioritaskan istilah yang mirip dengan Kamus Referensi di atas.
    - Kembalikan HANYA format JSON yang valid.
    
    Teks Laporan untuk Dianalisis:
    ---
    {text_to_analyze}
    ---
    """
    try:
        response = model.generate_content(prompt_template)
        json_output = response.text.replace("```json", "").replace("```", "").strip()
        print("NER berhasil diekstrak dengan panduan KB.")
        return json_output
    except Exception as e:
        print(f"Error API Gemini saat NER: {e}")
        return None