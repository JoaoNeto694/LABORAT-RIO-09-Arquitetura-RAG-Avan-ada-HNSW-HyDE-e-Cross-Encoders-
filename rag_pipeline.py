import time
import numpy as np
import faiss
from google.colab import userdata
from groq import Groq

# Configurações
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Hiperparâmetros HNSW
HNSW_M = 16 # nº máximo de conexões por nó em cada camada
HNSW_EF_CONSTRUCTION = 64 # qualidade da construção
HNSW_EF_SEARCH = 32 # qualidade da busca

# PASSO 1 — Indexação HNSW
def construir_indice_hnsw(corpus, embedder):
    
    # Vetoriza o corpus e constrói um índice HNSW com FAISS.
    # Usa similaridade de cosseno (via produto interno em vetores normalizados).
    embeddings = embedder.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(embeddings)

    return index

# PASSO 2 — Query Transformation com HyDE
# HyDE: pede ao LLM para 'alucinar' um trecho técnico que responderia à query.
def gerar_documento_hipotetico(query):
    prompt = (
        "Você é um médico especialista. O usuário fez a pergunta abaixo em linguagem coloquial. "
        "Escreva um parágrafo curto (3 a 5 frases) que pareceria um trecho de manual médico técnico "
        "respondendo a essa pergunta, usando jargão clínico apropriado (nomes de síndromes, sintomas "
        "em terminologia médica, possíveis tratamentos). Não diga que é hipotético — escreva como "
        "se fosse o próprio manual.\n\n"
        f"Pergunta do usuário: {query}\n\n"
        "Trecho do manual:"
    )

    # Usa a Groq API
    client = Groq(api_key=userdata.get('GROQ_API_KEY'))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()
