"""
NURU V3 - Assistant IA personnel local ultra-optimisé.

Architecture complète pour Apple Silicon M1 8 Go.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

from src.config import config
from src.intent_classifier import IntentClassifier, IntentType
from src.rag_engine import RAGEngine
from src.embedder import Embedder
from src.llm_local import LLMLocal
from src.llm_cloud import LLMCloud
from src.memory_store import MemoryStore
from src.stt import STT
from src.tts import TTS


logger = logging.getLogger(__name__)


class NuruCore:
    """
    Cœur de l'assistant NURU V3.

    Pipeline complet : Input → STT → Intent → Routing → LLM → Output → TTS
    Optimisé pour faible latence et empreinte mémoire minimale.
    """

    def __init__(self):
        self.intent_classifier: Optional[IntentClassifier] = None
        self.rag_engine: Optional[RAGEngine] = None
        self.embedder: Optional[Embedder] = None
        self.llm_local: Optional[LLMLocal] = None
        self.llm_cloud: Optional[LLMCloud] = None
        self.memory: Optional[MemoryStore] = None
        self.stt: Optional[STT] = None
        self.tts: Optional[TTS] = None

        self._initialized = False
        self._use_nano_model = False

    async def initialize(self) -> None:
        """Initialiser tous les composants avec lazy loading."""
        if self._initialized:
            return

        logger.info("Initialisation de NURU V3...")

        # Initialiser le classifier d'intents (ultra rapide)
        self.intent_classifier = IntentClassifier()
        await self.intent_classifier.initialize()

        # Initialiser l'embedder (seulement si nécessaire)
        self.embedder = Embedder()
        await self.embedder.initialize()

        # Initialiser le moteur RAG
        self.rag_engine = RAGEngine(self.embedder)
        await self.rag_engine.initialize()

        # Initialiser les LLM locaux
        self.llm_local = LLMLocal()
        await self.llm_local.initialize()

        # Initialiser le client cloud
        self.llm_cloud = LLMCloud()

        # Initialiser la mémoire
        self.memory = MemoryStore()
        await self.memory.initialize()

        # Audio - lazy loading uniquement
        self.stt = STT()
        self.tts = TTS()

        self._initialized = True
        logger.info("NURU V3 prêt.")

    async def shutdown(self) -> None:
        """Libérer proprement toutes les ressources."""
        logger.info("Arrêt de NURU V3...")

        if self.tts:
            await self.tts.shutdown()

        if self.stt:
            await self.stt.shutdown()

        if self.memory:
            await self.memory.shutdown()

        if self.llm_local:
            await self.llm_local.unload()

        if self.rag_engine:
            await self.rag_engine.shutdown()

        if self.embedder:
            await self.embedder.unload()

        if self.intent_classifier:
            await self.intent_classifier.shutdown()

        self._initialized = False
        logger.info("NURU V3 arrêté.")

    async def process_query(
        self,
        query: str,
        use_audio_input: bool = False,
        use_audio_output: bool = False,
    ) -> AsyncGenerator[str, None]:
        """
        Traiter une requête utilisateur complète.

        Args:
            query: Texte ou chemin vers fichier audio
            use_audio_input: Si True, query est un chemin audio
            use_audio_output: Si True, générer sortie audio

        Yields:
            Tokens de réponse en streaming
        """
        if not self._initialized:
            await self.initialize()

        start_time = asyncio.get_event_loop().time()

        try:
            # Étape 1: STT si input audio
            if use_audio_input and self.stt:
                logger.debug("Transcription audio...")
                query = await self.stt.transcribe(query)
                if not query.strip():
                    yield "[Erreur: Audio non reconnu]"
                    return

            # Étape 2: Classification d'intent (< 20ms)
            assert self.intent_classifier is not None
            intent = await self.intent_classifier.classify(query)
            logger.info(f"Intent: {intent.type.value} (confidence: {intent.confidence:.2f})")

            # Étape 3: Routing intelligent
            response_generator = await self._route_query(query, intent)

            # Étape 4: Streaming des tokens avec buffer TTS
            token_buffer = []
            sentence_buffer = []

            async for token in response_generator:
                yield token
                token_buffer.append(token)
                sentence_buffer.append(token)

                # Détection de fin de phrase pour TTS
                if any(token.endswith(p) for p in [".", "!", "?", "\n"]):
                    if use_audio_output and self.tts and sentence_buffer:
                        sentence = "".join(sentence_buffer).strip()
                        if len(sentence) > 10:
                            await self.tts.speak_buffered(sentence)
                    sentence_buffer = []

            # Flush final TTS
            if use_audio_output and self.tts and sentence_buffer:
                sentence = "".join(sentence_buffer).strip()
                if len(sentence) > 10:
                    await self.tts.speak_buffered(sentence)

            # Sauvegarder en mémoire
            if self.memory and token_buffer:
                await self.memory.add_interaction(query, "".join(token_buffer))

            elapsed = asyncio.get_event_loop().time() - start_time
            logger.info(f"Requête traitée en {elapsed * 1000:.0f}ms")

        except Exception as e:
            logger.error(f"Erreur lors du traitement: {e}", exc_info=True)
            yield f"[Erreur: {str(e)}]"

    async def _route_query(
        self,
        query: str,
        intent: IntentType,
    ) -> AsyncGenerator[str, None]:
        """
        Router la requête vers le modèle approprié.

        SIMPLE → Nano 1.5B
        MEDIUM/RAG → Standard 3B + RAG
        COMPLEX → Cloud DeepSeek V4
        """
        assert self.llm_local is not None
        assert self.rag_engine is not None
        assert self.llm_cloud is not None

        # Vérifier état mémoire pour swap intelligent
        await self._check_memory_state()

        if intent.type == "SIMPLE":
            # Requêtes simples → Nano modèle
            logger.debug("Routing: NANO (1.5B)")
            async for token in self.llm_local.generate(
                query,
                model="nano",
                system_prompt=self._get_system_prompt("simple"),
            ):
                yield token

        elif intent.type == "MEDIUM":
            # Requêtes moyennes → RAG + Standard
            logger.debug("Routing: STANDARD (3B) + RAG")

            # Recherche RAG hybride
            contexts = await self.rag_engine.search(query, top_k=config.RAG_TOP_K)

            if contexts and contexts[0].score >= config.RAG_CONFIDENCE_THRESHOLD:
                # Information trouvée → utiliser RAG
                context_text = "\n\n".join([c.text for c in contexts])
                augmented_query = self._build_rag_query(query, context_text)

                async for token in self.llm_local.generate(
                    augmented_query,
                    model="standard",
                    system_prompt=self._get_system_prompt("rag"),
                ):
                    yield token
            else:
                # Pas assez confiant → fallback sans RAG
                logger.warning("RAG confidence too low, falling back to standard")
                async for token in self.llm_local.generate(
                    query,
                    model="standard",
                    system_prompt=self._get_system_prompt("medium"),
                ):
                    yield token

        else:  # COMPLEX
            # Requêtes complexes → Cloud
            logger.debug("Routing: CLOUD (DeepSeek V4)")

            # Vérifier si API key configurée
            if not config.DEEPSEEK_API_KEY:
                logger.warning("DeepSeek API key missing, falling back to local")
                async for token in self.llm_local.generate(
                    query,
                    model="standard",
                    system_prompt=self._get_system_prompt("complex"),
                ):
                    yield token
                return

            async for token in self.llm_cloud.generate(
                query,
                system_prompt=self._get_system_prompt("complex"),
            ):
                yield token

    async def _check_memory_state(self) -> None:
        """Vérifier l'état mémoire et ajuster les modèles."""
        import psutil

        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        if available_gb < config.RAM_THRESHOLD_CRITICAL_GB:
            # Mode critique: décharger embedder
            if self.embedder and not self.embedder.is_unloaded():
                logger.warning(f"RAM critique ({available_gb:.2f}GB), déchargement embedder")
                await self.embedder.unload()
            self._use_nano_model = True

        elif available_gb < config.RAM_THRESHOLD_LOW_GB:
            # Mode économie: utiliser nano par défaut
            self._use_nano_model = True

        else:
            self._use_nano_model = False

    def _get_system_prompt(self, mode: str) -> str:
        """Obtenir le prompt système selon le mode."""
        base_prompt = """Tu es NURU, un assistant IA personnel intelligent et précis.

RÈGLES STRICTES:
1. Ne jamais inventer d'informations
2. Citer systématiquement les sources quand disponibles
3. Distinguer clairement connaissances générales et documents fournis
4. Être concis et direct
5. Répondre dans la langue de l'utilisateur

"""
        if mode == "simple":
            return base_prompt + "Mode: Réponses courtes et directes."

        elif mode == "rag":
            return (
                base_prompt
                + """
Mode: RAG activé.
Utilise EXCLUSIVEMENT les informations fournies dans le contexte ci-dessous.
Si l'information n'est pas dans le contexte, dis-le clairement.
Format de citation: [Source: nom_du_document]

"""
            )

        elif mode == "medium":
            return base_prompt + "Mode: Réponses équilibrées avec explications."

        else:  # complex
            return (
                base_prompt
                + """
Mode: Raisonnement approfondi.
Décomposer les problèmes complexes étape par étape.
Vérifier la cohérence logique avant de répondre.

"""
            )

    def _build_rag_query(self, query: str, context: str) -> str:
        """Construire la requête augmentée avec contexte RAG."""
        return f"""Contexte document:
{context}

Question: {query}

Réponse basée sur le contexte:"""

    async def ingest_document(self, file_path: str) -> int:
        """
        Ingérer un document dans la base RAG.

        Args:
            file_path: Chemin vers le fichier (txt, md, pdf)

        Returns:
            Nombre de chunks ingérés
        """
        from src.ingestion import DocumentIngester

        if not self._initialized:
            await self.initialize()

        ingester = DocumentIngester(self.rag_engine)
        count = await ingester.ingest(file_path)

        logger.info(f"Document ingéré: {file_path} ({count} chunks)")
        return count


# Instance singleton
nuru_core: Optional[NuruCore] = None


async def get_nuru() -> NuruCore:
    """Obtenir l'instance singleton de NuruCore."""
    global nuru_core
    if nuru_core is None:
        nuru_core = NuruCore()
        await nuru_core.initialize()
    return nuru_core
