# NURU V3 - Architecture Technique Détaillée

## Vue d'ensemble

NURU V3 est un assistant IA personnel conçu pour fonctionner localement sur MacBook M1 8 Go avec une empreinte mémoire minimale (< 4.5 Go) et une latence ultra-faible (TTFT < 800ms).

---

## Principes de Conception

### 1. Sobriété Mémoire Absolue

- **Aucun module inutile** : Chaque dépendance doit être justifiée
- **Lazy loading agressif** : STT, TTS, modèles chargés uniquement si nécessaire
- **Déchargement explicite** : `mx.metal.clear_cache()` + `gc.collect()` après usage
- **Swap intelligent** : Si RAM < 2.5GB → nano modèle ; Si RAM < 1.5GB → décharger embedder

### 2. Pipeline Async Ultra-Rapide

```
Input → STT (lazy) → Intent (<20ms) → Routing → LLM → Stream → TTS (buffer)
```

- **Zéro blocking I/O** : Tout en async sauf SQLite
- **Streaming immédiat** : Tokens affichés dès génération
- **Buffer TTS par phrase** : Lecture fluide sans attendre fin réponse

### 3. Anti-Hallucination Agressive

- **Confidence Gate RAG** : Score < 0.35 → "Information non trouvée" (sans appeler LLM)
- **Routing pré-inférence** : Décision AVANT génération
- **Prompt système strict** : Interdiction d'invention + citations obligatoires
- **Cache exact MD5** : TTL 300s, pas de cache sémantique complexe

---

## Stack Technique Justifiée

### LLM Local - MLX

**Pourquoi MLX ?**
- Natif Apple Silicon (Metal)
- Pas de PyTorch (économie ~2GB RAM)
- Streaming tokens natif
- Quantification 4-bit intégrée

**Modèles :**
- Nano: `DeepSeek-R1-Distill-Qwen-1.5B-4bit` (~600MB RAM, TTFT ~400ms)
- Standard: `Qwen2.5-3B-Instruct-4bit` (~2.5GB RAM, TTFT ~800ms)

### Embeddings - MLX Embedders

**Pourquoi `mlx-community/multilingual-e5-small-mlx` ?**
- 384 dimensions (suffisant pour RAG)
- Multilingue (FR/EN/ES/etc.)
- ~120MB RAM
- Ultra-rapide sur M1

### Vector DB - sqlite-vec

**Pourquoi sqlite-vec ?**
- Zéro serveur (contrairement à ChromaDB, Qdrant, Weaviate)
- Extension SQLite légère
- Index vectoriel natif
- Combine avec FTS5 pour BM25

**Architecture hybride :**
```
Requête → Embedding (384D)
        ↓
Vector Search (cosine similarity) + BM25 (FTS5)
        ↓
Score hybride: 0.7 * vector + 0.3 * BM25
        ↓
Top-K results triés
```

### Intent Classification - fastText + Regex

**Pourquoi pas un LLM ?**
- fastText: < 5ms vs ~500ms pour un petit LLM
- Moins de 50MB RAM
- Précision suffisante pour routing simple

**Patterns regex prioritaires :**
- Salutations → SIMPLE
- Remerciements → SIMPLE
- "Résume", "Analyse", "Crée code" → COMPLEX
- Autres → fastText fallback

### Audio - Whisper + Piper

**STT: mlx-whisper**
- Modèle tiny par défaut (~100MB)
- Transcription dans executor thread (non-blocant)
- Libération immédiate après usage

**TTS: Piper**
- Synthèse locale rapide
- Bufferisation par phrases
- Playback async avec queue

### UI - PySide6 Overlay

**Pourquoi PySide6 et pas web ?**
- Zéro serveur HTTP (économie FastAPI + uvicorn)
- Overlay natif macOS
- Animations fluides (QPropertyAnimation)
- Style cyberpunk via CSS-like stylesheets

---

## Gestion Mémoire Détaillée

### Budget RAM Cible

| Composant | RAM | Notes |
|-----------|-----|-------|
| Base (Python + deps) | ~500 MB | |
| Intent Classifier | ~50 MB | fastText |
| Embedder | ~120 MB | Chargé si RAG nécessaire |
| LLM Nano | ~600 MB | Toujours chargé |
| LLM Standard | ~2.5 GB | Chargé si intent MEDIUM |
| RAG Engine | ~200 MB | sqlite-vec + index |
| STT (Whisper) | ~300 MB | Lazy, libéré après |
| TTS (Piper) | ~100 MB | Lazy |
| UI PySide6 | ~150 MB | |
| **Total baseline** | **~3.9 GB** | Sans audio |
| **Total peak audio** | **~4.4 GB** | Avec STT+TTS |

### Stratégie de Swap

```python
async def _check_memory_state(self):
    available_gb = psutil.virtual_memory().available / (1024**3)

    if available_gb < 1.5:  # Critique
        await self.embedder.unload()  # Libère ~120MB
        use_nano_only = True

    elif available_gb < 2.5:  # Économie
        use_nano_only = True  # Pas de modèle standard
```

---

## Pipeline Complet

### 1. Réception Input

```python
async def process_query(query, use_audio_input=False, use_audio_output=False):
    # STT si audio
    if use_audio_input:
        query = await stt.transcribe(audio_path)
```

### 2. Classification Intent

```python
intent = await classifier.classify(query)
# < 20ms, regex ou fastText
```

### 3. Routing

```python
if intent == SIMPLE:
    model = "nano"  # 1.5B
elif intent == MEDIUM:
    contexts = await rag.search(query)
    if contexts[0].score >= 0.35:
        # RAG activé
        query = build_rag_query(query, contexts)
    model = "standard"  # 3B
else:  # COMPLEX
    if DEEPSEEK_API_KEY:
        return cloud.generate(query)  # Fallback API
    model = "standard"
```

### 4. Génération Streaming

```python
async for token in llm.generate(prompt, model):
    yield token  # Affichage immédiat
    if sentence_end(token):
        await tts.speak_buffered(sentence)  # TTS async
```

---

## Tests et Validation

### Tests Unitaires

- `test_intent_classifier.py` : Vérifie regex + fastText
- `test_rag_engine.py` : Vérifie hybrid search
- `test_memory_store.py` : Vérifie cache TTL
- `test_chunking.py` : Vérifie découpage documents

### Benchmarks Cibles

| Métrique | Cible | Mesure |
|----------|-------|--------|
| Intent classification | < 20ms | `time.classify()` |
| Embedding batch (5) | < 100ms | `time.embed()` |
| RAG search | < 200ms | `time.search()` |
| TTFT SIMPLE | < 800ms | `time.first_token` |
| TTFT MEDIUM+RAG | < 1.5s | |
| RAM baseline | < 4.3GB | `psutil.virtual_memory()` |

---

## Scripts et Déploiement

### Installation

```bash
./scripts/install.sh
```

Crée :
- Environnement virtuel `.venv`
- Installe dépendances
- Télécharge modèles MLX
- Configure `.env`

### Téléchargement Modèles

```bash
python scripts/download_models.py
```

Télécharge :
- Nano 1.5B (~1GB disk)
- Standard 3B (~2GB disk)
- Embedder (~200MB disk)
- Whisper tiny (~100MB disk)

---

## Maintenance et Évolutivité

### Code Quality

- **Type hints** : 100% typé, compatible mypy strict
- **Docstrings** : Courtes, informatives
- **Logging structuré** : Pas de print(), rotation 10MB
- **Async propre** : Aucun time.sleep(), queues asyncio

### Modules Indépendants

Chaque module peut être testé isolément :
- `IntentClassifier` : Mockable sans MLX
- `RAGEngine` : Testable avec mock embedder
- `LLMLocal` : Testable avec petits modèles

### Évolutions Futures Possibles

1. **LanceDB** : Si sqlite-vec limité (mais plus lourd)
2. **Modèles personnalisés** : Fine-tuning Qwen sur données utilisateur
3. **Plugins** : Système d'extensions pour outils externes
4. **Synchro cloud** : Backup chiffré conversations (opt-in)

---

## Conclusion

NURU V3 démontre qu'un assistant IA local **vraiment utilisable** est possible sur MacBook M1 8 Go avec :

- **Fluidité** : TTFT < 800ms, streaming immédiat
- **Sobriété** : < 4.5GB RAM, zéro swap
- **Fiabilité** : RAG hybride, anti-hallucination
- **Maintenabilité** : < 3000 lignes, 13 modules, 100% typé

L'architecture privilégie systématiquement **la simplicité à la complexité**, **la performance à la fonctionnalité**, et **l'utilisateur à la technologie**.
