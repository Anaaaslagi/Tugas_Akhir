import os
import google.generativeai as genai

def setup_gemini():
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("="*50)
        print("ERROR: API KEY TIDAK DITEMUKAN.")
        print("Pastikan Anda telah mengatur 'GOOGLE_API_KEY' di Environment Variables.")
        print("PENTING: Anda HARUS RESTART terminal/VS Code setelah mengaturnya.")
        print("="*50)
        exit()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        print("API Key berhasil dimuat dari Environment Variable.")
        return model
    except Exception as e:
        print(f"Error saat konfigurasi API: {e}")
        exit()

# Inisialisasi model agar bisa di-import oleh file lain
model = setup_gemini()