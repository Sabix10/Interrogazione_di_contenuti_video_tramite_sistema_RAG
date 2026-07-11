# 🎓 Interrogazione di contenuti video tramite sistema RAG

![Python](https://img.shields.io/badge/Python-3.9%2B-1B365D?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?style=for-the-badge&logo=flask&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-red?style=for-the-badge)
![LM Studio](https://img.shields.io/badge/LM%20Studio-Local%20LLM-27AE60?style=for-the-badge)
![Ragas](https://img.shields.io/badge/Ragas-Evaluation-orange?style=for-the-badge)

**Progetto di Natural Language Processing (A.A. 2025/2026)**  
**Autori:** Sabato Malafronte, Antonio Di Lauro  

---

## 📌 Panoramica del Progetto

Questo progetto implementa un'architettura **RAG (Retrieval-Augmented Generation)** avanzata, progettata per estrarre, indicizzare e interrogare il contenuto di videolezioni e video di YouTube attraverso un'interfaccia web intuitiva. 

Il sistema supera i limiti del tradizionale chunking a caratteri fissi adottando una **vettorizzazione semantica dinamica**: il testo viene segmentato in tempo reale riconoscendo i cambi di argomento basati sulla distanza euclidea e sulla similarità del coseno degli embedding. L'interferenza e la generazione delle risposte sono affidate a Large Language Models (LLM) eseguiti interamente in locale tramite **LM Studio**, garantendo privacy e assenza di costi di API esterne.

---

## ✨ Funzionalità Chiave

* **Estrazione Sottotitoli con Fallback:** Utilizzo di `youtube-transcript-api` per scaricare le tracce in italiano, con passaggio automatico alla lingua inglese o alla prima lingua disponibile in caso di assenza dei sottotitoli locali.
* **Chunking Semantico Dinamico:** La segmentazione del testo avviene valutando la similarità tra frasi consecutive tokenizzate tramite NLTK. È impostata una soglia di similarità ottimizzata a `0.78` con un freno d'emergenza di massimo `12` frasi per blocco.
* **Database Vettoriale FAISS:** Gli embedding sono generati tramite il modello `intfloat/multilingual-e5-base` e indicizzati in memoria con struttura `IndexFlatL2` per una ricerca semantica ultra-rapida.
* **Filtro di Tolleranza L2:** Il sistema scarta automaticamente i chunk recuperati che superano una soglia di distanza euclidea pari a `0.415`, impedendo all'LLM di ricevere contesto irrilevante.
* **Doppia Modalità di Conversazione:**
  * 👨‍🏫 **Assistente Standard:** Risposte dirette, rigorose e basate esclusivamente sul contesto (Temperatura: `0.1`).
  * 🏛️ **Dialogo Socratico:** Il modello guida lo studente al ragionamento critico senza fornire risposte preconfezionate (Temperatura: `0.4`).
* **Gestione Multi-Video:** Interfaccia web dotata di menu a scorrimento per isolare indici FAISS e storici di conversazione (ultimi 4 messaggi) in base al video selezionato.
* **Valutazione e Stress Testing:** Integrazione nativa del framework **RAGAS** per misurare *Faithfulness* e *Answer Relevancy*, con esportazione automatica dei report in fogli Excel multi-scheda.

---

## 🛠️ Stack Tecnologico e Architettura

| Componente | Tecnologia / Libreria | Dettaglio / Modello |
| :--- | :--- | :--- |
| **Backend Web** | Flask | Routing, gestione sessioni e API REST |
| **NLP & Tokenization** | NLTK | Moduli `punkt` e `punkt_tab` per sentence tokenization |
| **Embedding Model** | Sentence Transformers | `intfloat/multilingual-e5-base` (vettori densi) |
| **Vector Database** | FAISS CPU | Ricerca di similarità L2 (`IndexFlatL2`) |
| **LLM Inference** | LM Studio / OpenAI Client | Server locale su porta `localhost:1234` |
| **Modelli LLM Testati** | Llama / Qwen / Gemma | `meta-llama-3.1-8b-instruct`, `qwen2.5-1.5b-instruct` |
| **Evaluation Framework** | RAGAS & Langchain | Valutazione real-time e batch con LLM-as-a-Judge |
| **Frontend** | HTML5, CSS3, JS Vanilla | Interfaccia reattiva con animazioni Skeleton e Spinner |

---

## 📁 Struttura della Repository

```text
├── app.py # Server backend Flask, logica RAG e gestione DB FAISS
├── requirements.txt # Dipendenze esatte del progetto
└── tests/
.   ├── grid_search.py # Algoritmo di ottimizzazione empirica delle soglie di chunking
.   ├── test_ragas.py # Script di stress-test multi-video ed esportazione Excel
.   └── dataset_ragas_multi_video.json # Dataset di benchmarking (Telegiornali, Lezioni, Doc)
└── templates/
    └── index.html # Interfaccia utente web
