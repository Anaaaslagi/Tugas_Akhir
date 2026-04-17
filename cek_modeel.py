import os
import google.generativeai as genai

# Konfigurasi API Key
api_key = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("Mencari model Embedding yang tersedia untuk API Key Anda...\n")

tersedia = False
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(f"✅ Ditemukan: {m.name}")
        tersedia = True

if not tersedia:
    print("❌ Tidak ada model embedding yang tersedia untuk API Key ini.")