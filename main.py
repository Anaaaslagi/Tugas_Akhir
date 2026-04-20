from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted
import json
import os
import time
from document_processor import select_pdf_file, extract_text_from_pdf, parse_report_sections
from ner_module import extract_entities_with_gemini

# IMPORT FUNGSI BARU DI SINI
from rag_module import retrieve_similar_case, generate_client_summary_rag, generate_client_summary_baseline

def main():
    # --- 0. MUAT KNOWLEDGE BASE ---
    print("Memuat Knowledge Base...")
    try:
        with open("kb_ner_mcmi.json", "r", encoding="utf-8") as f:
            ner_kb = json.load(f)
        with open("kb_rag_mcmi.json", "r", encoding="utf-8") as f:
            rag_kb = json.load(f)
    except FileNotFoundError:
        print("ERROR: File Knowledge Base tidak ditemukan! Pastikan kb_ner_mcmi.json dan kb_rag_mcmi.json ada di folder yang sama.")
        return

    # --- 1. PEMROSESAN PDF ---
    pdf_path = select_pdf_file()
    if not pdf_path: return
    
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text: return

    structured_report = parse_report_sections(raw_text)
    if not structured_report: return

    # --- 2. PIPELINE NER ---
    text_for_ner = (
        structured_report.get('hasil_pemeriksaan', '') + "\n" +
        structured_report.get('dinamika_psikologis', '') + "\n" +
        structured_report.get('tantangan_pemberdayaan', '')
    )
    
    extracted_entities = extract_entities_with_gemini(text_for_ner, ner_kb)
    if not extracted_entities: return

    # --- SIMPAN HASIL NER KE FILE ---
    print("\nMenyimpan hasil NER ke file...")
    base_filename = os.path.basename(pdf_path) 
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_json_path = f"{filename_without_ext}_NER.json"

    try:
        ner_data = json.loads(extracted_entities)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(ner_data, f, ensure_ascii=False, indent=4)
        print(f"BERHASIL: Hasil NER telah disimpan ke -> {output_json_path}")
    except json.JSONDecodeError:
        print(f"PERINGATAN: Gagal mem-parsing JSON dari output NER. Menyimpan sebagai file teks mentah.")
        output_txt_path = f"{filename_without_ext}_NER_raw.txt"
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(extracted_entities)
        print(f"BERHASIL: Output mentah NER disimpan ke -> {output_txt_path}")

    # Jeda sejenak untuk menghindari limit API sebelum masuk tahap generasi
    print("\nMenunggu 5 detik sebelum tahap generasi teks...")
    time.sleep(5)

    # --- 3. GENERASI BASELINE (TANPA RAG) ---
    recommendation_text = structured_report.get('saran_rekomendasi', '')
    
    print("\nMenjalankan model BASELINE (Tanpa Knowledge Base)...")
    summary_baseline = generate_client_summary_baseline(extracted_entities, recommendation_text)

    # --- 4. PIPELINE RAG (DENGAN KNOWLEDGE BASE) ---
    print("\nMenjalankan model RAG (Dengan Knowledge Base)...")
    retrieved_context = retrieve_similar_case(extracted_entities, rag_kb)
    summary_rag = generate_client_summary_rag(extracted_entities, recommendation_text, retrieved_context)
    
    # --- 5. CETAK PERBANDINGAN ---
    if summary_baseline and summary_rag:
        print("\n" + "="*60)
        print(" " * 15 + "HASIL GENERASI LLM BASELINE (TANPA RAG)")
        print("="*60)
        print(summary_baseline)
        
        print("\n\n" + "="*60)
        print(" " * 15 + "HASIL GENERASI LLM + RAG (DENGAN REFERENSI KASUS)")
        print("="*60)
        print(summary_rag)
        print("="*60 + "\n")

        # Opsional: Simpan hasil perbandingan ke dalam file teks
        output_summary_path = f"{filename_without_ext}_Perbandingan_Ringkasan.txt"
        with open(output_summary_path, 'w', encoding='utf-8') as f:
            f.write("=== HASIL GENERASI BASELINE (TANPA RAG) ===\n")
            f.write(summary_baseline + "\n\n")
            f.write("=== HASIL GENERASI LLM + RAG ===\n")
            f.write(summary_rag + "\n")
        print(f"File perbandingan berhasil disimpan ke -> {output_summary_path}")

if __name__ == "__main__":
    main()