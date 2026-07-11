from flask import Flask, request, jsonify, render_template
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from nltk.tokenize import sent_tokenize
from urllib.parse import urlparse, parse_qs
import nltk
import urllib.request
import json
from openai import OpenAI

# RAGAS
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig

# SCARICHIAMO IL MODELLO PUNKT PER TOKENIZATION
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# CONFIGURIAMO IL CLIENT LOCALE
app = Flask(__name__)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# --- MODELLI LLM OGGETTO DEI TEST ---

# MODEL_NAME = "qwen2.5-1.5b-instruct"
MODEL_NAME = "meta-llama-3.1-8b-instruct"
# MODEL_NAME = "gemma-2-9b-it"

video_database = {}

# --- CONFIGURAZIONI RAG ---
CHUNK_SIMILARITY_THRESHOLD = 0.78  # Soglia per il cambio argomento
MAX_SENTENCES_PER_CHUNK = 12       # Freno d'emergenza: max frasi in un blocco
RETRIEVAL_MAX_K = 6                # Limite massimo di chunk da estrarre
RETRIEVAL_DISTANCE_THRESHOLD = 0.415 # Soglia di tolleranza per scartare chunk irrilevanti
# --------------------------

SYSTEM_PROMPT = """Sei un assistente universitario rigoroso.
Il tuo compito è rispondere alle domande dell'utente basandoti ESCLUSIVAMENTE sul blocco ### INIZIO CONTESTO ###.

REGOLA ASSOLUTA: 
È severamente vietato utilizzare le tue conoscenze personali pre-addestrate. 
Se il contesto fornito non contiene la risposta esatta e completa alla domanda, devi rispondere ESATTAMENTE e SOLO con la seguente frase:
"Non lo so."

Rispondi in lingua italiana, in modo chiaro, dettagliato ma conciso, e non ripetere mai il testo del contesto nella risposta.
"""

# SYSTEM PROMPT NEL CASO L'UTENTE SCELGA LA MODALITA' CHAT SOCRATICA
SOCRATIC_PROMPT = """
Sei un docente universitario che usa rigorosamente il METODO SOCRATICO.
Il tuo obiettivo è guidare lo studente a comprendere il concetto da solo, senza MAI dare la risposta diretta o spiegazioni preconfezionate.
Devi rispondere utilizzando esclusivamente
le informazioni presenti nel contesto.
Regole:
- Non usare conoscenze esterne.
- Non inventare informazioni.
- Se la risposta non è contenuta nel contesto scrivi:
Non lo so.
Rispondi sempre in italiano.
Fornisci spiegazioni dettagliate ma concise.
"""

embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")

# --- MODELLO LLM SELEZIONATO COME GIUDICE RAGAS ---
JUDGE_BASE_URL = "http://localhost:1234/v1"
JUDGE_MODEL_NAME = "meta-llama-3.1-8b-instruct"

# --- INIZIALIZZAZIONE COMPONENTI DI GIUDIZIO RAGAS ---
ragas_llm = LangchainLLMWrapper(ChatOpenAI(
    base_url=JUDGE_BASE_URL,
    api_key="lm-studio",
    model=JUDGE_MODEL_NAME,
    timeout=240,
    max_retries=2
))

ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
)

faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm
answer_relevancy.embeddings = ragas_embeddings



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def get_video_title(video_url):
    try:
        url = f"https://www.youtube.com/oembed?url={video_url}&format=json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data.get("title", "Video senza titolo")
    except Exception as e:
        print(f"Impossibile recuperare il titolo: {e}")
        return "Video YouTube (Titolo non recuperabile)"


# ESTRAZIONE ID DEL VIDEO
def get_video_id_from_url(youtube_url):
    query = urlparse(youtube_url)

    if query.hostname in ['www.youtube.com', 'youtube.com']:

        if 'v' in query.query:
            return parse_qs(query.query)['v'][0]

        path = query.path.split('/')
        return path[-1]

    elif query.hostname == 'youtu.be':

        return query.path[1:]

    return None


# ESTRAZIONE TRASCRIZIONE DEL VIDEO
def get_video_transcript(video_url):
    video_id = get_video_id_from_url(video_url)
    yt_api = YouTubeTranscriptApi()

    try:
        transcript = yt_api.fetch(video_id, languages=['it'])
        print(f"Sottotitoli in italiano recuperati per il video: {video_id}")
    except Exception:
        try:
            transcript_list = yt_api.list_transcripts(video_id)
            transcript = transcript_list.find_transcript(['it', 'en']).fetch()
            print(f"Sottotitoli (Fallback Italiano/Inglese) recuperati per il video: {video_id}")
        except Exception as e:
            raise ValueError(
                f"Impossibile recuperare i sottotitoli per questo video. Verifica che siano abilitati su YouTube. Errore: {str(e)}")

    text = " ".join([row.text for row in transcript])
    return text

# CREAZIONE DEI CHUNKS (DINAMICA E SEMANTICA)
def create_chunks(text, similarity_threshold=CHUNK_SIMILARITY_THRESHOLD, max_sentences=MAX_SENTENCES_PER_CHUNK):
    sentences = sent_tokenize(text, language='italian')

    if not sentences:
        return []

    print(f"\n--- INIZIO CHUNKING SEMANTICO ---")
    print(f"Totale frasi da analizzare: {len(sentences)}")

    sentence_embeddings = embedding_model.encode(sentences, convert_to_numpy=True)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(sentence_embeddings[i], sentence_embeddings[i - 1])

        # Se l'argomento è lo stesso e non abbiamo superato il limite massimo di frasi
        if sim >= similarity_threshold and len(current_chunk) < max_sentences:
            current_chunk.append(sentences[i])
        else:
            # Tagliamo se cambia argomento oppure se il chunk è diventato troppo lungo
            if len(current_chunk) >= max_sentences:
                print(f"-> AZIONE: Limite di {max_sentences} frasi raggiunto! Freno d'emergenza attivato.")
            else:
                print("-> AZIONE: Cambio argomento rilevato! Chiudo il chunk.")

            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"\n--- FINE CHUNKING ---")
    print(f"Totale chunk dinamici creati: {len(chunks)}\n")

    return chunks

# CREAZIONE EMBEDDINGS PER DATABASE
def create_db_from_youtube_video_url(video_url):
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        raise ValueError("URL YouTube non valido.")

    # Recuperiamo il titolo del video
    title = get_video_title(video_url)

    # Controllo Cache applicativa
    if video_id in video_database:
        print(f"\n[DATABASE] Il video '{title}' è già presente in memoria.")
        app.config["current_video_id"] = video_id
        return video_id, title

    # Ingestion standard
    transcript = get_video_transcript(video_url)
    chunks = create_chunks(transcript)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    # Salviamo includendo il titolo della videolezione
    video_database[video_id] = {
        "title": title,
        "transcript": transcript,
        "chunks": chunks,
        "faiss_index": index
    }

    app.config["current_video_id"] = video_id
    print(f"[DATABASE] Archiviato con successo: '{title}'")

    return video_id, title

# RECUPERO DEI CHUNK DINAMICO CON FILTRO DI DISTANZA L2
def get_similarity_from_query(query, max_k=RETRIEVAL_MAX_K, distance_threshold=RETRIEVAL_DISTANCE_THRESHOLD, video_id=None):
    # Se il front-end non passa un ID, usare l'ultimo video elaborato
    if video_id is None:
        video_id = app.config.get("current_video_id")

    # Controllo di sicurezza sul dizionario globale che mantiene le informazioni sulle videolezioni
    if not video_id or video_id not in video_database:
        raise ValueError("DB vettoriale non creato o video non trovato. Elabora prima un video.")

    # Estrazione dei dati isolati per la videolezione specifica
    archive_data = video_database[video_id]
    index = archive_data["faiss_index"]
    chunks = archive_data["chunks"]

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_embedding.astype(np.float32), max_k)

    retrieved_chunks = []

    print(f"\n--- RICERCA DINAMICA NELL'ARCHIVIO (Video ID: {video_id}) PER: '{query}' ---")

    for i, dist in zip(indices[0], distances[0]):
        if dist <= distance_threshold:
            print(f"Chunk {i} | Distanza: {dist:.4f} -> ACCETTATO")
            retrieved_chunks.append(chunks[i])
        else:
            print(f"Chunk {i} | Distanza: {dist:.4f} -> SCARTATO (Supera {distance_threshold})")

    print(f"Totale chunk dinamici recuperati: {len(retrieved_chunks)} / {max_k}\n")

    return retrieved_chunks


def run_ragas_evaluation(question, answer, contexts):
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts]
    }
    dataset = Dataset.from_dict(data)

    try:
        print("\n[RAGAS] Calcolo delle metriche di valutazione in corso...")

        configurazione_run = RunConfig(timeout=300)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            run_config=configurazione_run  # <--- INIETTA QUI LA CONFIGURAZIONE
        )

        faith_score = np.mean(result['faithfulness'])
        rel_score = np.mean(result['answer_relevancy'])

        print("\n=============================================")
        print("REPORT DI VALUTAZIONE RAGAS (Local Judge)")
        print(f"FAITHFULNESS: {faith_score:.4f}")
        print(f"ANSWER RELEVANCY: {rel_score:.4f}")
        print("=============================================\n")
        return result
    except Exception as e:
        import traceback
        print(f"Impossibile completare la valutazione RAGAS: {e}")
        traceback.print_exc()
        return None


# INTERFACCIA HTML
@app.route('/')
def index():
    return render_template('index.html')


# ROUTE QUANDO L'UTENTE PREME INVIA DOPO AVER INSERITO IL LINK DEL VIDEO YOUTUBE
@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    url = data.get('url')
    if not url:
        return jsonify({'error': 'URL mancante'}), 400
    try:
        video_id, title = create_db_from_youtube_video_url(url)
        return jsonify({
            'message': 'Video elaborato e indicizzato con successo.',
            'video_id': video_id,
            'title': title
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ROUTE PER LA RICERCA DEI CHUNKS
@app.route(
    '/search',
    methods=['POST']
)
def search():
    data = request.json
    query = data.get('query')

    k = data.get('k', 3)

    if not query:
        return jsonify(
            {
                'error': 'Query mancante'
            }
        ), 400

    try:
        results = get_similarity_from_query(query, max_k=int(k))
        return jsonify(
            {
                'results': results
            }
        )
    except Exception as e:
        return jsonify(
            {
                'error': str(e)
            }
        ), 500


chat_history = {}

# ROUTE QUANDO L'UTENTE EFFETTUA UNA DOMANDA AL LLM
@app.route('/chatbot', methods=['POST'])
def chatbot():
    session_id = request.form.get('session_id', 'default_user')
    user_query = request.form['query']
    mode = request.form.get('mode', 'standard')

    video_id = request.form.get('video_id')

    # Inizializzazione storico
    if session_id not in chat_history:
        chat_history[session_id] = []

    current_system_prompt = SOCRATIC_PROMPT if mode == 'socratic' else SYSTEM_PROMPT

    try:
        retrieved_chunks = get_similarity_from_query(user_query, video_id=video_id)
    except ValueError as e:
        return "Errore: Seleziona o elabora una videolezione prima di fare una domanda."

    if not retrieved_chunks:
        bot_reply = "Non lo so."
        chat_history[session_id].append({"role": "user", "content": user_query})
        chat_history[session_id].append({"role": "assistant", "content": bot_reply})
        return bot_reply
    # --------------------------

    retrieved_context = "\n\n".join(retrieved_chunks)

    print(f"\n--- CONTESTO INVIATO AL LLM ---\n{retrieved_context}\n-------------------------------\n")

    augmented_prompt = (
        f"Usa le informazioni contenute nel seguente contesto per rispondere alla domanda finale.\n\n"
        f"### INIZIO CONTESTO ###\n"
        f"{retrieved_context}\n"
        f"### FINE CONTESTO ###\n\n"
        f"DOMANDA: {user_query}"
    )

    messages = [{"role": "system", "content": current_system_prompt}]
    messages.extend(chat_history[session_id][-4:])
    messages.append({"role": "user", "content": augmented_prompt})

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.4 if mode == 'socratic' else 0.1,
            max_tokens=600
        )
        bot_reply = response.choices[0].message.content

    except Exception as e:
        print(f"Errore durante la generazione LLM: {e}")
        return "Si è verificato un errore di connessione con il modello locale. Controlla il terminale."

    chat_history[session_id].append({"role": "user", "content": user_query})
    chat_history[session_id].append({"role": "assistant", "content": bot_reply})

    # === VALUTAZIONE RAGAS IN REAL-TIME ===
    run_ragas_evaluation(user_query, bot_reply, retrieved_chunks)

    return bot_reply

if __name__ == '__main__':
    app.run(debug=True)