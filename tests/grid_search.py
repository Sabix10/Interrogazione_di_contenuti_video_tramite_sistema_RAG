from app import get_video_transcript, embedding_model, cosine_similarity
from nltk.tokenize import sent_tokenize

def run_threshold_grid_search(video_url):
    print(f"\n--- AVVIO GRID SEARCH ---")
    print(f"Recupero trascrizione per: {video_url}...")

    transcript = get_video_transcript(video_url)
    sentences = sent_tokenize(transcript, language='italian')

    if not sentences:
        print("Errore: Nessuna frase trovata nella trascrizione.")
        return

    print(f"Calcolo degli embedding per {len(sentences)} frasi in corso...")
    sentence_embeddings = embedding_model.encode(sentences, convert_to_numpy=True)

    thresholds = [0.75, 0.76, 0.77, 0.78, 0.79, 0.80]
    results = []

    print("Analisi delle soglie in corso...\n")

    for thresh in thresholds:
        chunks = []
        current_chunk = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i - 1])

            if sim >= thresh:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        avg_sentences = len(sentences) / len(chunks) if chunks else 0

        results.append({
            "threshold": thresh,
            "num_chunks": len(chunks),
            "avg_sentences": round(avg_sentences, 1)
        })

    print(f"Statistiche basate su un totale di {len(sentences)} frasi:")
    print("-" * 65)
    print(f"| {'Soglia (Threshold)':<18} | {'Numero di Chunk':<15} | {'Media Frasi/Chunk':<20} |")
    print("-" * 65)

    for res in results:
        t = f"{res['threshold']:.2f}"
        c = res['num_chunks']
        a = res['avg_sentences']
        print(f"| {t:<18} | {c:<15} | {a:<20} |")

    print("-" * 65)
    print("\n--- FINE GRID SEARCH ---")

if __name__ == '__main__':
    run_threshold_grid_search("https://www.youtube.com/watch?v=kIaXye4-7kE")