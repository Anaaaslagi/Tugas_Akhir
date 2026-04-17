import json
import tkinter as tk
from tkinter import filedialog

def select_json_file(title):
    """Membuka kotak dialog untuk memilih file JSON."""
    root = tk.Tk()
    root.withdraw()  # Sembunyikan jendela tkinter utama
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("JSON files", "*.json")]
    )
    
    if not file_path:
        print(f"Pemilihan file '{title}' dibatalkan.")
        return None
        
    print(f"File dipilih ({title}): {file_path}")
    return file_path

def load_json(filepath):
    """Memuat file JSON dari path."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File tidak ditemukan di {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: File {filepath} bukan JSON yang valid.")
        return None

def calculate_metrics(predicted_list, actual_list):
    """Menghitung TP, FP, FN berdasarkan dua list."""
    
    # Normalisasi: ubah ke huruf kecil dan hapus spasi di awal/akhir
    predicted_set = {str(item).strip().lower() for item in predicted_list}
    actual_set = {str(item).strip().lower() for item in actual_list}
    
    # Hitung True Positives, False Positives, dan False Negatives
    tp = len(predicted_set.intersection(actual_set))
    fp = len(predicted_set.difference(actual_set))
    fn = len(actual_set.difference(predicted_set))
    
    return tp, fp, fn

def evaluate_ner(predicted_file, actual_file):
    """Mengevaluasi file NER yang diprediksi terhadap file ground truth."""
    
    predicted_data = load_json(predicted_file)
    actual_data = load_json(actual_file)
    
    if not predicted_data or not actual_data:
        print("Evaluasi dibatalkan karena file error.")
        return

    all_metrics = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Mendapatkan semua kunci (kategori) dari kedua file
    all_categories = set(predicted_data.keys()) | set(actual_data.keys())

    for category in all_categories:
        predicted_list = predicted_data.get(category, [])
        actual_list = actual_data.get(category, [])
        
        tp, fp, fn = calculate_metrics(predicted_list, actual_list)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Hitung metrik per kategori (Precision, Recall, F1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        all_metrics[category] = {
            "Precision": f"{precision:.2%}",
            "Recall": f"{recall:.2%}",
            "F1-Score": f"{f1:.2%}",
            "TP": tp, # True Positive
            "FP": fp, # False Positive
            "FN": fn  # False Negative
        }

    # Hitung metrik keseluruhan (Macro Average)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("\n" + "="*40)
    print("--- HASIL EVALUASI NER PER KATEGORI ---")
    print("="*40)
    for category, metrics in all_metrics.items():
        print(f"\nKategori: {category}")
        print(f"  Precision: {metrics['Precision']} (TP: {metrics['TP']}, FP: {metrics['FP']})")
        print(f"  Recall:    {metrics['Recall']} (TP: {metrics['TP']}, FN: {metrics['FN']})")
        print(f"  F1-Score:  {metrics['F1-Score']}")
        
    print("\n" + "="*40)
    print("--- HASIL EVALUASI KESELURUHAN (MACRO) ---")
    print("="*40)
    print(f"Overall Precision: {overall_precision:.2%}")
    print(f"Overall Recall:    {overall_recall:.2%}")
    print(f"Overall F1-Score:  {overall_f1:.2%}")
    print("="*40)

# --- JALANKAN EVALUASI ---
if __name__ == "__main__":
    print("Memulai Skrip Evaluasi NER...")
    
    # 1. Minta pengguna memilih file Prediksi
    file_prediksi = select_json_file("Pilih File Prediksi JSON (..._NER.json)")
    if not file_prediksi:
        exit()
        
    # 2. Minta pengguna memilih file Ground Truth
    file_aktual = select_json_file("Pilih File Ground Truth JSON (Kunci Jawaban)")
    if not file_aktual:
        exit()

    # 3. Jalankan evaluasi
    evaluate_ner(file_prediksi, file_aktual)