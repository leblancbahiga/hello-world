"""
Tests unitaires pour NURU V3.

Couvre:
- Intent classifier
- RAG engine
- Cache MD5
- Chunking
"""

import pytest
import hashlib
from pathlib import Path


# =====================================================================
# TESTS INTENT CLASSIFIER
# =====================================================================

class TestIntentClassifier:
    """Tests pour le classifier d'intents."""

    @pytest.mark.asyncio
    async def test_regex_simple_greeting(self):
        """Les salutations doivent être classées SIMPLE."""
        from src.intent_classifier import IntentClassifier, IntentEnum

        classifier = IntentClassifier()
        await classifier.initialize()

        for greeting in ["bonjour", "salut", "hello", "hi", "coucou"]:
            result = await classifier.classify(greeting)
            assert result.type == IntentEnum.SIMPLE
            assert result.method == "regex"

    @pytest.mark.asyncio
    async def test_regex_simple_thanks(self):
        """Les remerciements doivent être classés SIMPLE."""
        from src.intent_classifier import IntentClassifier, IntentEnum

        classifier = IntentClassifier()
        await classifier.initialize()

        for thanks in ["merci", "thanks", "thank you"]:
            result = await classifier.classify(thanks)
            assert result.type == IntentEnum.SIMPLE

    @pytest.mark.asyncio
    async def test_regex_complex_summarize(self):
        """Les demandes de résumé doivent être classées COMPLEX."""
        from src.intent_classifier import IntentClassifier, IntentEnum

        classifier = IntentClassifier()
        await classifier.initialize()

        result = await classifier.classify("Résume ce document")
        assert result.type == IntentEnum.COMPLEX

    @pytest.mark.asyncio
    async def test_regex_complex_code(self):
        """Les demandes de code doivent être classées COMPLEX."""
        from src.intent_classifier import IntentClassifier, IntentEnum

        classifier = IntentClassifier()
        await classifier.initialize()

        result = await classifier.classify("Crée un programme Python")
        assert result.type == IntentEnum.COMPLEX

    @pytest.mark.asyncio
    async def test_fasttext_fallback(self):
        """Le fallback fastText doit fonctionner pour les requêtes non regex."""
        from src.intent_classifier import IntentClassifier, IntentEnum

        classifier = IntentClassifier()
        await classifier.initialize()

        # Requête qui ne matche aucun pattern regex
        result = await classifier.classify("Je cherche des informations sur la biologie")
        assert result.type in [IntentEnum.MEDIUM, IntentEnum.COMPLEX]
        assert result.method == "fasttext"


# =====================================================================
# TESTS RAG ENGINE
# =====================================================================

class TestRAGEngine:
    """Tests pour le moteur RAG."""

    @pytest.mark.asyncio
    async def test_chunk_creation(self):
        """La création de chunks doit fonctionner."""
        from src.rag_engine import Chunk

        chunk = Chunk(
            id="test-123",
            text="Ceci est un texte de test.",
            source="test.txt",
        )

        assert chunk.id == "test-123"
        assert chunk.text == "Ceci est un texte de test."
        assert chunk.source == "test.txt"

    @pytest.mark.asyncio
    async def test_hybrid_search_merge(self):
        """La fusion hybride vectoriel + BM25 doit fonctionner."""
        from src.rag_engine import RAGEngine, SearchResult
        from src.embedder import Embedder

        # Mock embedder
        class MockEmbedder:
            async def embed_single(self, text):
                import mlx.core as mx
                return mx.zeros(384)

        rag = RAGEngine(MockEmbedder())

        vector_results = [
            SearchResult(text="doc A", source="A", score=0.9, method="vector"),
            SearchResult(text="doc B", source="B", score=0.7, method="vector"),
        ]

        bm25_results = [
            SearchResult(text="doc B", source="B", score=0.8, method="bm25"),
            SearchResult(text="doc C", source="C", score=0.6, method="bm25"),
        ]

        merged = rag._merge_results(vector_results, bm25_results, top_k=3)

        # doc B devrait être premier (présent dans les deux)
        assert len(merged) <= 3
        assert any(r.text == "doc B" for r in merged)


# =====================================================================
# TESTS MEMORY STORE
# =====================================================================

class TestMemoryStore:
    """Tests pour le stockage mémoire."""

    def test_md5_hash(self):
        """Le hash MD5 doit être cohérent."""
        from src.memory_store import MemoryStore

        store = MemoryStore()
        text = "Hello World"
        hash1 = store._compute_hash(text)
        hash2 = store._compute_hash(text)

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 = 128 bits = 32 hex chars

    @pytest.mark.asyncio
    async def test_cache_ttl(self):
        """Le cache TTL doit expirer correctement."""
        from src.memory_store import MemoryStore
        import time

        store = MemoryStore()
        await store.initialize()

        # Mettre en cache
        await store.set_cached("query1", "response1", ttl=1)

        # Devrait être présent
        cached = await store.get_cached("query1")
        assert cached == "response1"

        # Attendre expiration
        await asyncio.sleep(1.5)

        # Devrait être expiré
        cached = await store.get_cached("query1")
        assert cached is None


# =====================================================================
# TESTS CHUNKING
# =====================================================================

class TestChunking:
    """Tests pour le découpage de documents."""

    def test_token_count_approximation(self):
        """L'approximation de tokens doit être raisonnable."""
        from src.ingestion import DocumentIngester
        from src.rag_engine import RAGEngine

        class MockRAG:
            embedder = None

        ingester = DocumentIngester(MockRAG())

        text = "Bonjour, ceci est un test de comptage de tokens."
        tokens = ingester._count_tokens(text)

        # ~50 chars / 4 ≈ 12-13 tokens
        assert 10 <= tokens <= 15

    def test_paragraph_splitting(self):
        """Le découpage de paragraphes longs doit fonctionner."""
        from src.ingestion import DocumentIngester
        from src.rag_engine import RAGEngine

        class MockRAG:
            embedder = None

        ingester = DocumentIngester(MockRAG())

        # Paragraphe long
        long_para = " ".join(["Phrase"] * 100)  # ~100 phrases
        chunks = ingester._split_paragraph(long_para, max_tokens=50)

        # Devrait créer plusieurs chunks
        assert len(chunks) > 1

        # Chaque chunk ne devrait pas dépasser trop la limite
        for chunk in chunks:
            tokens = ingester._count_tokens(chunk)
            assert tokens <= 100  # Marge acceptable


# =====================================================================
# TESTS CONFIG
# =====================================================================

class TestConfig:
    """Tests pour la configuration."""

    def test_config_paths_created(self):
        """Les chemins de configuration doivent exister."""
        from src.config import config

        assert config.DATA_DIR.exists()
        assert config.INDEXES_DIR.exists()
        assert config.MODELS_DIR.exists()
        assert config.LOGS_DIR.exists()

    def test_config_embedding_dim(self):
        """La dimension d'embedding doit être 384."""
        from src.config import config

        assert config.EMBEDDING_DIM == 384

    def test_config_rag_threshold(self):
        """Le seuil RAG doit être 0.35."""
        from src.config import config

        assert config.RAG_CONFIDENCE_THRESHOLD == 0.35


# =====================================================================
# RUN TESTS
# =====================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
