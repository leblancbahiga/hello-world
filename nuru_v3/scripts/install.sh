#!/bin/bash
# Script d'installation de NURU V3 pour macOS Apple Silicon

set -e

echo "=============================================="
echo "  NURU V3 - Installation"
echo "=============================================="

# Vérifier Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Vérifier version minimale (3.11)
REQUIRED_VERSION="3.11"
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "❌ Erreur: Python 3.11+ requis"
    exit 1
fi

# Créer environnement virtuel
echo ""
echo "📦 Création de l'environnement virtuel..."
python3 -m venv .venv

# Activer environnement
echo "🔌 Activation de l'environnement..."
source .venv/bin/activate

# Mettre à jour pip
echo "⬆️  Mise à jour de pip..."
pip install --upgrade pip

# Installer dépendances
echo "📥 Installation des dépendances..."
pip install -e ".[dev]"

# Télécharger modèles
echo "⬇️  Téléchargement des modèles..."
python scripts/download_models.py

# Configuration initiale
echo "⚙️  Configuration initiale..."
cat > .env << EOF
# API DeepSeek (optionnel)
DEEPSEEK_API_KEY=

# Logs
LOG_LEVEL=INFO

# Whisper
WHISPER_MODEL=tiny

# Voix Piper
PIPER_VOICE=fr_FR-siwis-medium
EOF

# Créer dossier data
mkdir -p data indexes models logs

echo ""
echo "=============================================="
echo "  ✅ Installation terminée !"
echo "=============================================="
echo ""
echo "Pour lancer NURU:"
echo "  source .venv/bin/activate"
echo "  python -m src.daemon"
echo ""
echo "Pour l'overlay UI:"
echo "  python -m src.overlay"
echo ""
echo "=============================================="
