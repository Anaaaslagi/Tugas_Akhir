import pdfplumber
import re
import tkinter as tk
from tkinter import filedialog

def select_pdf_file():
    print("Membuka kotak dialog pemilihan file...")
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela tkinter utama
    
    file_path = filedialog.askopenfilename(
        title="Pilih Laporan MCMI-IV PDF",
        filetypes=[("PDF files", "*.pdf")]
    )
    
    if not file_path:
        print("Pemilihan file dibatalkan.")
        return None
        
    print(f"File dipilih: {file_path}")
    return file_path

def extract_text_from_pdf(pdf_path):
    print(f"Mengekstrak teks dari {pdf_path}...")
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
        print("Ekstraksi teks berhasil.")
        return full_text
    except Exception as e:
        print(f"Error membaca PDF {pdf_path}: {e}")
        return None

def parse_report_sections(raw_text):
    print("Memilah teks laporan menjadi beberapa bagian...")
    sections = {}
    
    match_1 = re.search(r"IV\.\s*Hasil Pemeriksaan\s*(.*?)(?=\s*V\.\s*Dinamika Psikologis)", raw_text, re.DOTALL | re.IGNORECASE)
    if match_1: sections['hasil_pemeriksaan'] = match_1.group(1).strip()

    match_2 = re.search(r"V\.\s*Dinamika Psikologis\s*(.*?)(?=\s*VI\.\s*Tantangan)", raw_text, re.DOTALL | re.IGNORECASE)
    if match_2: sections['dinamika_psikologis'] = match_2.group(1).strip()

    match_3 = re.search(r"VI\.\s*Tantangan.*?\s*(.*?)(?=\s*VII\.\s*Saran)", raw_text, re.DOTALL | re.IGNORECASE)
    if match_3: sections['tantangan_pemberdayaan'] = match_3.group(1).strip()

    match_4 = re.search(r"VII\.\s*Saran.*?\s*(.*?)(?=\s*Demikian hasil pemeriksaan|$)", raw_text, re.DOTALL | re.IGNORECASE)
    if match_4: sections['saran_rekomendasi'] = match_4.group(1).strip()
        
    print("\n--- STATUS PARSING ---")
    print(f"Hasil Pemeriksaan: {'Berhasil' if 'hasil_pemeriksaan' in sections else 'GAGAL'}")
    print(f"Dinamika Psikologis: {'Berhasil' if 'dinamika_psikologis' in sections else 'GAGAL'}")
    print(f"Tantangan & Pemberdayaan: {'Berhasil' if 'tantangan_pemberdayaan' in sections else 'GAGAL'}")
    print(f"Saran & Rekomendasi: {'Berhasil' if 'saran_rekomendasi' in sections else 'GAGAL'}")
    
    if len(sections) == 4:
        print("Semua bagian laporan berhasil dipilah!")
        return sections
    else:
        print("\nError: Ada bagian yang masih gagal diekstrak. Proses dibatalkan.")
        return None