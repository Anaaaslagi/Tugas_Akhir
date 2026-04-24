from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json

def evaluate_generation():
    print("="*60)
    print("📊 EVALUASI MODEL GENERASI (ROUGE & BLEU)")
    print("="*60)

    # 1. SIAPKAN TEKS REFERENSI (Ground Truth)
    # Ini adalah ringkasan yang ditulis/divalidasi oleh Psikolog secara manual
    referensi_pakar = """
    Klien menunjukkan indikasi kecemasan sosial dan afek yang datar. 
    Sebagai faktor pelindung, klien didukung oleh keluarga yang kuat serta wawasan diri yang memadai. 
    Intervensi yang direkomendasikan adalah terapi kognitif perilaku (CBT) 
    untuk menurunkan tingkat resistensi klien selama proses terapi.
    """

    # 2. SIAPKAN TEKS PREDIKSI (Hasil Generasi AI-mu)
    # Ambil teks ini dari hasil run main.py (bagian Summary RAG)
    hasil_ai = """
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

    # Membersihkan teks (menghapus spasi berlebih dan enter)
    ref_text = " ".join(referensi_pakar.strip().split())
    gen_text = " ".join(hasil_ai.strip().split())

    # --- MENGHITUNG ROUGE ---
    # rouge1: kecocokan per kata (unigram)
    # rouge2: kecocokan per dua kata berurutan (bigram)
    # rougeL: kecocokan urutan kalimat terpanjang (Longest Common Subsequence)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    scores = scorer.score(ref_text, gen_text)

    # --- MENGHITUNG BLEU ---
    # BLEU butuh input dalam bentuk list token (kata)
    ref_tokens = [ref_text.lower().split()] # Perhatikan kurung sikunya (list of lists untuk referensi)
    gen_tokens = gen_text.lower().split()

    # Smoothing digunakan agar skor tidak langsung 0 jika tidak ada frasa panjang yang cocok persis
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)

    # --- MENAMPILKAN HASIL ---
    print("[1] ROUGE SCORE (Skala 0.0 - 1.0)")
    print("-" * 30)
    print(f"ROUGE-1 (Unigram) : F1 = {scores['rouge1'].fmeasure:.4f} | Precision = {scores['rouge1'].precision:.4f} | Recall = {scores['rouge1'].recall:.4f}")
    print(f"ROUGE-2 (Bigram)  : F1 = {scores['rouge2'].fmeasure:.4f} | Precision = {scores['rouge2'].precision:.4f} | Recall = {scores['rouge2'].recall:.4f}")
    print(f"ROUGE-L (LCS)     : F1 = {scores['rougeL'].fmeasure:.4f} | Precision = {scores['rougeL'].precision:.4f} | Recall = {scores['rougeL'].recall:.4f}")
    
    print("\n[2] BLEU SCORE (Skala 0.0 - 1.0)")
    print("-" * 30)
    print(f"Skor BLEU         : {bleu_score:.4f}")
    print("="*60)

if __name__ == "__main__":
    evaluate_generation()