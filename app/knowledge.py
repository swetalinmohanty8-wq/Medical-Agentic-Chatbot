from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from groq import Groq
from app.config import settings

@lru_cache()
def clients():
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise ValueError("API keys for Groq and Pinecone must be set in the environment variables.")
    return Groq(api_key=settings.groq_api_key), Pinecone(api_key=settings.pinecone_api_key)


def ensure_index():
    _, pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(name=settings.pinecone_index, dimension=512, metric='cosine', spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    return pc.Index(settings.pinecone_index)


def embed(texts):
    model = SentenceTransformer(settings.embed_model)
    return model.encode(texts).tolist()


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text',''), 'score': round(m.score,3), 'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]
    