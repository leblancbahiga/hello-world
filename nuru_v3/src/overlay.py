"""
Overlay PySide6 cyberpunk pour NURU V3.

Interface ultra-légère:
- Fenêtre translucide
- Streaming temps réel
- Animations subtiles
- Design cyberpunk minimaliste
"""

import logging
import sys
from typing import Optional

from PySide6.QtCore import (
    Qt,
    QTimer,
    Signal,
    Slot,
    QPropertyAnimation,
    QEasingCurve,
)
from PySide6.QtGui import (
    QFont,
    QFontDatabase,
    QColor,
    QPalette,
    QPainter,
    QPainterPath,
    QBrush,
    QPen,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QFrame,
    QGraphicsDropShadowEffect,
)

from src.config import config


logger = logging.getLogger(__name__)


class CyberpunkOverlay(QMainWindow):
    """
    Fenêtre overlay style cyberpunk.

    Features:
    - Translucidité
    - Bords néon
    - Typographie monospace
    - Animations fluides
    """

    sig_stream_token = Signal(str)
    sig_response_ready = Signal(str)

    def __init__(self):
        super().__init__()

        self._current_response = ""
        self._is_streaming = False

        self._setup_ui()
        self._setup_styles()
        self._setup_animations()

    def _setup_ui(self) -> None:
        """Configurer l'interface."""
        # Fenêtre sans cadre, translucide
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Dimensions
        self.setGeometry(
            100,
            100,
            config.UI_WIDTH,
            config.UI_HEIGHT,
        )

        # Widget central
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Header
        self.header_label = QLabel("NURU V3")
        self.header_label.setObjectName("headerLabel")
        layout.addWidget(self.header_label)

        # Zone de texte principale (réponses)
        self.response_area = QTextEdit()
        self.response_area.setReadOnly(True)
        self.response_area.setObjectName("responseArea")
        layout.addWidget(self.response_area)

        # Input utilisateur
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(0, 0, 0, 0)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Posez votre question...")
        self.input_field.setObjectName("inputField")
        self.input_field.returnPressed.connect(self._on_submit)
        input_layout.addWidget(self.input_field)

        self.submit_btn = QPushButton("ENVOYER")
        self.submit_btn.setObjectName("submitBtn")
        self.submit_btn.clicked.connect(self._on_submit)
        input_layout.addWidget(self.submit_btn)

        layout.addWidget(input_frame)

        # Status bar discrète
        self.status_label = QLabel("Prêt")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

    def _setup_styles(self) -> None:
        """Configurer le style cyberpunk."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {config.UI_BG_COLOR};
            }}

            QWidget {{
                font-family: "{config.UI_FONT_FAMILY}", "Courier New", monospace;
                color: {config.UI_PRIMARY_COLOR};
            }}

            #headerLabel {{
                font-size: 24px;
                font-weight: bold;
                color: {config.UI_SECONDARY_COLOR};
                padding: 10px;
                border-bottom: 2px solid {config.UI_PRIMARY_COLOR};
            }}

            #responseArea {{
                background-color: rgba(10, 10, 15, 0.8);
                border: 1px solid {config.UI_PRIMARY_COLOR};
                border-radius: 8px;
                padding: 15px;
                font-size: 14px;
                line-height: 1.6;
            }}

            #inputFrame {{
                background-color: rgba(10, 10, 15, 0.6);
                border: 1px solid {config.UI_SECONDARY_COLOR};
                border-radius: 8px;
                padding: 10px;
            }}

            #inputField {{
                background-color: rgba(0, 0, 0, 0.5);
                border: 1px solid {config.UI_PRIMARY_COLOR};
                border-radius: 4px;
                padding: 10px;
                color: {config.UI_PRIMARY_COLOR};
                font-size: 14px;
            }}

            #inputField:focus {{
                border: 1px solid {config.UI_SECONDARY_COLOR};
            }}

            #submitBtn {{
                background-color: {config.UI_PRIMARY_COLOR};
                color: {config.UI_BG_COLOR};
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
            }}

            #submitBtn:hover {{
                background-color: {config.UI_SECONDARY_COLOR};
            }}

            #statusLabel {{
                font-size: 11px;
                color: rgba(0, 255, 255, 0.6);
                padding: 5px;
            }}
        """)

    def _setup_animations(self) -> None:
        """Configurer animations."""
        # Animation d'apparition
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(config.UI_OPACITY)

    def show_animated(self) -> None:
        """Afficher avec animation."""
        self.show()
        self.fade_animation.start()

    @Slot(str)
    def stream_token(self, token: str) -> None:
        """Ajouter un token au flux de réponse."""
        self._current_response += token
        self.response_area.setPlainText(self._current_response)

        # Scroll automatique vers le bas
        scrollbar = self.response_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

        self._is_streaming = True
        self.status_label.setText("Génération en cours...")

    @Slot(str)
    def set_response(self, response: str) -> None:
        """Définir la réponse complète."""
        self._current_response = response
        self.response_area.setPlainText(response)
        self._is_streaming = False
        self.status_label.setText("Prêt")

    def _on_submit(self) -> None:
        """Gérer soumission utilisateur."""
        query = self.input_field.text().strip()

        if not query or self._is_streaming:
            return

        logger.debug(f"Input utilisateur: {query}")

        # Clear input
        self.input_field.clear()

        # Clear previous response
        self._current_response = ""
        self.response_area.clear()

        # Émettre signal pour traitement
        self.sig_response_ready.emit(query)

    def paintEvent(self, event) -> None:
        """Dessiner bordure néon."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Créer path arrondi
        path = QPainterPath()
        path.addRoundedRect(
            self.rect().adjusted(5, 5, -5, -5),
            15,
            15,
        )

        # Bordure néon
        pen = QPen(QColor(config.UI_PRIMARY_COLOR), 2)
        painter.setPen(pen)
        painter.drawPath(path)

    def keyPressEvent(self, event) -> None:
        """Gérer touches clavier."""
        if event.key() == Qt.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)


def run_overlay():
    """Lancer l'overlay."""
    app = QApplication(sys.argv)

    # Charger polices
    QFontDatabase.addApplicationFontFromFamily("SF Mono")

    overlay = CyberpunkOverlay()
    overlay.show_animated()

    sys.exit(app.exec())


if __name__ == "__main__":
    run_overlay()
