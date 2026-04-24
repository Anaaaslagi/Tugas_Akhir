import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from config import model # Memanggil konfigurasi API dari filemu

def ragas_llm_judge():
    print("="*60)
    print("⚖️ EVALUASI RAGAS (LLM-as-a-Judge) DIMULAI")
    print("="*60)

    # 1. SIAPKAN DATA (Diambil dari hasil run main.py)
    # --------------------------------------------------
    # Prompt asli yang kamu perintahkan ke sistem RAG
    pertanyaan = """
    Anda adalah seorang asisten psikolog yang suportif.
    Tugas Anda menulis ringkasan laporan psikologis yang mudah dipahami klien.
    
    Instruksi:
    - Jelaskan temuan utama secara suportif.
    - Sebutkan kekuatan klien.
    - Berikan rekomendasi langkah-langkah secara eksklusif berdasarkan 'Rencana Intervensi' yang diberikan.
    - JANGAN mengulang teks mentah, buatlah ringkasan naratif yang natural.
    """
    
    # Konteks (Gabungan NER pasien + Referensi KB yang ditarik)
    konteks_diberikan = """
    ENTITAS: Gejala: Cemas, Afek Datar. Kekuatan: Dukungan keluarga baik.
    REFERENSI KB: Kasus serupa menunjukkan CBT bertahap efektif untuk klien dengan kecemasan tinggi.
    """
    
    # Hasil akhir yang di-generate oleh modelmu (Summary RAG)
    jawaban_ai = """
    Halo, terima kasih sudah berbagi dan mempercayakan kami untuk membantu Anda memahami diri lebih baik. Dari laporan yang kami susun, ada beberapa hal menarik yang bisa kita lihat bersama.
    Kami melihat bahwa saat ini Anda mungkin sering merasa cemas ringan, khawatir berlebihan, dan kadang sulit untuk rileks. Pikiran Anda cenderung memikirkan berbagai kemungkinan buruk, yang bisa membuat Anda merasa gelisah dan tegang. Ada kalanya suasana hati Anda juga cukup mudah berubah, dan Anda mungkin merasa dorongan untuk bertindak impulsif atau tergesa-gesa, yang berisiko membuat pilihan yang kurang matang. Kami juga memahami bahwa Anda memiliki kebutuhan yang kuat untuk diperhatikan dan diakui, serta terkadang merasa ragu terhadap nilai diri Anda sendiri. Hal ini bisa memicu perfeksionisme dan standar diri yang sangat tinggi, serta membuat Anda mudah merasa tersinggung atau kecewa.
    Namun, di balik tantangan ini, Anda memiliki banyak kekuatan luar biasa yang menjadi modal berharga. Anda adalah pribadi yang sangat percaya diri pada kemampuan dan ide-ide Anda, mampu berpikir luas, dan berani membayangkan rencana-rencana besar. Semangat dan energi Anda tinggi, membuat Anda produktif dan mampu menginspirasi orang lain. Anda juga dikenal sebagai individu yang disiplin, teratur, bertanggung jawab, dan dapat diandalkan. Kemampuan Anda untuk membangun kesan menyenangkan, ceria, dan mudah bergaul juga menjadi daya tarik tersendiri. Keyakinan pada diri sendiri ini membuat Anda tidak mudah merasa kecil dan berani mengambil peluang.
    Untuk membantu Anda merasa lebih tenang dan stabil, ada beberapa langkah yang bisa kita coba bersama:
    1.  **Menerima Diri dan Situasi:** Belajar untuk memberi ruang bagi ketidaksempurnaan, baik pada diri sendiri maupun dalam situasi. Tidak semua hal harus sempurna atau sesuai rencana. Menerima bahwa kesalahan adalah bagian dari proses belajar dan tidak mengurangi nilai diri Anda.
    2.  **Berpikir Sebelum Bertindak:** Biasakan untuk membuat jeda sejenak sebelum mengambil keputusan penting atau bereaksi. Tanyakan pada diri sendiri: apakah ini langkah terbaik atau hanya dorongan sesaat? Ini akan membantu Anda mengendalikan tindakan yang tergesa-gesa dan membuat pilihan yang lebih matang.
    3.  **Membangun Hubungan yang Lebih Dalam:** Latih diri untuk lebih mendengarkan orang lain tanpa langsung membandingkan atau mempertahankan diri. Hubungan yang sehat dan setara akan membuat hidup Anda lebih stabil dan mengurangi rasa gelisah.
    4.  **Menemukan Ketenangan Internal:** Cari kegiatan yang benar-benar membantu menenangkan pikiran Anda, seperti olahraga teratur, menulis, atau aktivitas yang membuat Anda merasa rileks tanpa harus selalu tampil unggul atau mencari pengakuan. Ini juga bisa menjadi cara untuk menyalurkan kebutuhan Anda untuk tampil dan dihargai melalui kegiatan positif. Penting juga untuk mengembangkan rasa percaya diri yang bersumber dari pencapaian nyata Anda, bukan hanya dari pujian atau perhatian orang lain.
    Dukungan dari orang-orang terdekat juga sangat penting. Mereka bisa membantu dengan memberikan apresiasi tulus atas usaha dan tanggung jawab Anda, serta mendukung Anda untuk melihat nilai diri tidak hanya dari pencapaian, tetapi juga dari kedekatan dan kerja sama yang hangat. Dengan pendekatan yang konsisten dan suportif, kami yakin Anda dapat berkembang menjadi pribadi yang lebih kuat, stabil, dan tenang dalam menghadapi berbagai tantangan hidup.

    """
    # --------------------------------------------------

    print("Meminta Gemini untuk mengevaluasi sebagai Juri Pakar...")

    # Prompt khusus untuk menyuruh LLM menjadi Juri (RAGAS-style)
    prompt_juri = f"""
    Anda adalah juri penilai sistem AI di bidang Psikologi Klinis. 
    Tugas Anda adalah mengevaluasi kualitas jawaban sistem RAG berdasarkan metrik RAGAS.
    
    Data Evaluasi:
    [PERTANYAAN/INSTRUKSI]: {pertanyaan}
    [KONTEKS YANG DIBERIKAN (NER & KB)]: {konteks_diberikan}
    [JAWABAN YANG DIHASILKAN SISTEM]: {jawaban_ai}

    Beri nilai 1 hingga 10 untuk tiga kriteria berikut:
    1. FAITHFULNESS: Apakah JAWABAN murni didasarkan pada KONTEKS? (Skor rendah jika ada halusinasi/info ngarang).
    2. ANSWER RELEVANCE: Apakah JAWABAN menjawab PERTANYAAN dengan tepat dan gaya bahasa yang sesuai?
    3. CONTEXT RELEVANCE: Apakah KONTEKS yang diberikan berisi informasi yang relevan untuk menjawab pertanyaan?

    Kembalikan HANYA format JSON valid seperti ini:
    {{
      "faithfulness_score": 8,
      "answer_relevance_score": 9,
      "context_relevance_score": 7,
      "alasan_singkat": "..."
    }}
    """

    # Memaksa model menjawab dalam bentuk JSON
    generation_config = GenerationConfig(temperature=0.0, response_mime_type="application/json")
    
    try:
        response = model.generate_content(prompt_juri, generation_config=generation_config)
        hasil_penilaian = response.text
        
        print("\n[HASIL PENILAIAN RAGAS]")
        print("-" * 60)
        print(hasil_penilaian)
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error saat melakukan evaluasi: {e}")

if __name__ == "__main__":
    ragas_llm_judge()