"""
Classification d'intents ultra-rapide avec fastText.

Objectif: < 20ms par classification.
Aucun LLM utilisé pour le routing.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import fasttext

from src.config import config


logger = logging.getLogger(__name__)


class IntentEnum(str, Enum):
    """Types d'intents supportés."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


@dataclass
class IntentType:
    """Résultat de classification."""

    type: IntentEnum
    confidence: float
    method: str  # "regex" ou "fasttext"


# Patterns regex pour override rapide (priorité absolue)
REGEX_PATTERNS = {
    IntentEnum.SIMPLE: [
        r"^(bonjour|salut|coucou|hello|hi)\s*[!?.]*$",
        r"^(merci|thanks|thank you)\s*[!?.]*$",
        r"^(au revoir|bye|goodbye|à bientôt)\s*[!?.]*$",
        r"^combien (font|font|sont) \d+\s*[\+\-\*/]\s*\d+\s*[?]?$",
        r"^quel(le)? est (l'|la |le )?(temps|date|jour)",
        r"^qui (es|êtes)-?(tu|vous)\??$",
        r"^que (peux|pouvez)-(tu|vous) faire",
        r"^aide(-| )moi",
    ],
    IntentEnum.COMPLEX: [
        r"^(résume|résumez|resume) (ce|cet|cette|le|la|les)",
        r"^(analyse|analysez) (ce|cet|cette|le|la|les)",
        r"^(compare|comparez) .* (et|avec|à) .*",
        r"^(explique|expliquez) .* (en détail|profondément|complètement)",
        r"^(crée|créer|génère|générer) (un|une|du|de la) (code|programme|script)",
        r"^(écris|écrire|rédige|rédiger) (un|une) (lettre|email|message|texte)",
        r"^(traduis|traduire) .* en (anglais|français|espagnol|allemand|italien)",
        r"^(développe|développer|construis|construire) (un|une) (application|site|outil)",
    ],
}


class IntentClassifier:
    """
    Classifieur d'intents hybride: regex + fastText.

    Optimisé pour Apple Silicon avec chargement lazy.
    """

    def __init__(self):
        self.model: Optional[fasttext._FastText] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialiser le modèle fastText."""
        if self._initialized:
            return

        logger.debug("Initialisation du classifier d'intents...")

        # Vérifier si modèle existe
        model_path = config.INTENT_MODEL_PATH

        if model_path.exists():
            # Charger modèle existant
            self.model = fasttext.load_model(str(model_path))
            logger.info(f"Modèle fastText chargé: {model_path}")
        else:
            # Créer modèle par défaut (entraînement minimal)
            await self._train_default_model()

        self._initialized = True
        logger.debug("Classifier prêt.")

    async def _train_default_model(self) -> None:
        """Entraîner un modèle minimal par défaut."""
        import tempfile
        import os

        logger.info("Entraînement du modèle fastText par défaut...")

        # Données d'entraînement minimales
        training_data = [
            # SIMPLE
            ("bonjour", IntentEnum.SIMPLE),
            ("salut ça va", IntentEnum.SIMPLE),
            ("merci beaucoup", IntentEnum.SIMPLE),
            ("au revoir", IntentEnum.SIMPLE),
            ("quelle heure est-il", IntentEnum.SIMPLE),
            ("quel jour sommes-nous", IntentEnum.SIMPLE),
            ("qui es-tu", IntentEnum.SIMPLE),
            ("peux-tu m'aider", IntentEnum.SIMPLE),
            ("comment ça marche", IntentEnum.SIMPLE),
            ("c'est quoi ça", IntentEnum.SIMPLE),
            # MEDIUM
            ("qu'est-ce que c'est", IntentEnum.MEDIUM),
            ("comment faire pour", IntentEnum.MEDIUM),
            ("où puis-je trouver", IntentEnum.MEDIUM),
            ("pourquoi il y a", IntentEnum.MEDIUM),
            ("explique-moi", IntentEnum.MEDIUM),
            ("parle-moi de", IntentEnum.MEDIUM),
            ("donne-moi des informations sur", IntentEnum.MEDIUM),
            ("cherche dans mes documents", IntentEnum.MEDIUM),
            ("trouve des infos sur", IntentEnum.MEDIUM),
            # COMPLEX
            ("résume ce document", IntentEnum.COMPLEX),
            ("analyse cette situation", IntentEnum.COMPLEX),
            ("compare les deux approches", IntentEnum.COMPLEX),
            ("explique en détail", IntentEnum.COMPLEX),
            ("crée un programme python", IntentEnum.COMPLEX),
            ("écris une lettre formelle", IntentEnum.COMPLEX),
            ("traduis ce texte en anglais", IntentEnum.COMPLEX),
            ("développe une application", IntentEnum.COMPLEX),
        ]

        # Créer fichier temporaire pour fastText
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            for text, intent in training_data:
                f.write(f"__label__{intent.value} {text}\n")
            temp_path = f.name

        try:
            # Entraîner modèle
            self.model = fasttext.train_supervised(
                input=temp_path,
                epoch=25,
                lr=0.5,
                wordNgrams=2,
                dim=50,
                loss="softmax",
                verbose=0,
            )

            # Sauvegarder modèle
            config.INTENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(config.INTENT_MODEL_PATH))
            logger.info(f"Modèle sauvegardé: {config.INTENT_MODEL_PATH}")

        finally:
            # Nettoyer fichier temporaire
            os.unlink(temp_path)

    async def classify(self, text: str) -> IntentType:
        """
        Classifier une requête utilisateur.

        Args:
            text: Texte à classifier

        Returns:
            IntentType avec type et confiance
        """
        if not self._initialized:
            await self.initialize()

        text_lower = text.lower().strip()

        # Étape 1: Vérifier patterns regex (ultra rapide < 5ms)
        for intent, patterns in REGEX_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, text_lower):
                    logger.debug(f"Intent détecté par regex: {intent.value}")
                    return IntentType(type=intent, confidence=0.95, method="regex")

        # Étape 2: Fallback fastText (< 20ms)
        assert self.model is not None
        prediction = self.model.predict(text_lower.replace("\n", " "))

        label = prediction[0][0].replace("__label__", "")
        confidence = float(prediction[1][0])

        intent = IntentEnum(label)
        logger.debug(f"Intent détecté par fastText: {intent.value} ({confidence:.2f})")

        return IntentType(type=intent, confidence=confidence, method="fasttext")

    async def shutdown(self) -> None:
        """Libérer le modèle."""
        self.model = None
        self._initialized = False
        logger.debug("Classifier arrêté.")
