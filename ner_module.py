import json
from config import model # Memanggil model dari config.py

def extract_entities_with_gemini(text_to_analyze, ner_kb):
    print("Memulai tugas NER dengan Gemini + Knowledge Base...")
    
    entity_definitions = """
    - "GEJALA": Tanda atau simptom klinis yang dilaporkan (misal: afek datar, insomnia).
    - "DIAGNOSIS": Pola kepribadian atau sindrom klinis yang teridentifikasi (misal: Skizoid, Depresi Mayor).
    - "TANTANGAN": Faktor internal atau eksternal yang menghambat perkembangan klien (misal: resistensi terapi, lingkungan toksik).
    - "KEKUATAN": Faktor positif atau area pemberdayaan yang dimiliki klien (misal: dukungan keluarga, wawasan diri yang baik).
    """

    prompt_template = f"""
    Anda adalah asisten psikologi klinis yang ahli dalam Named Entity Recognition (NER).
    
    Tugas Anda adalah mengekstrak entitas dari teks laporan psikologis berdasarkan definisi berikut:
    {entity_definitions}

    MODUL PEMBELAJARAN (In-Context Learning):
    Pelajari pola ekstraksi dari referensi knowledge base berikut. Ini adalah contoh, BUKAN batasan mutlak:
    {json.dumps(ner_kb, indent=2)}

    Aturan Ekstraksi (SANGAT PENTING):
    1. Ekstraksi Dinamis: Anda TIDAK dibatasi hanya pada istilah yang ada di dalam Modul Pembelajaran. Jika Anda menemukan entitas baru di dalam teks yang relevan dan sesuai dengan definisi di atas, Anda WAJIB mengekstraknya.
    2. Akurasi Klinis: Pastikan istilah yang diekstrak tetap relevan dengan konteks evaluasi psikologis klinis.
    3. Format Output: Kembalikan HANYA format JSON list murni tanpa markdown atau teks pengantar.

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