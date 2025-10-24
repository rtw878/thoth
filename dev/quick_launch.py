#!/usr/bin/env python3
"""
Quick Launch Script for Thoth

This script launches the GUI without preloading models for faster startup.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt


class QuickLaunchWindow(QMainWindow):
    """Quick launch window for Thoth."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thoth - Quick Launch")
        self.setGeometry(100, 100, 600, 400)
        self._apply_branding()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Logo (prefer SVG cropped primary lockup)
        logo_path = self._resolve_brand_icon_path()
        if logo_path and logo_path.exists():
            if logo_path.suffix.lower() == ".svg":
                svg = QSvgWidget(str(logo_path))
                svg.setFixedHeight(56)
                layout.addWidget(svg)
            else:
                logo_label = QLabel()
                pix = QPixmap(str(logo_path)).scaledToHeight(56, Qt.TransformationMode.SmoothTransformation)
                logo_label.setPixmap(pix)
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(logo_label)
        
        # Welcome message
        welcome_label = QLabel("<h1>Thoth - Historical Handwriting Recognition</h1>")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)
        
        info_label = QLabel(
            "<p>This is a quick launch version. For full functionality:</p>"
            "<ul>"
            "<li>Launch full app: <code>python src/app/main.py</code></li>"
            "<li>Test functionality: <code>python test_functionality.py</code></li>"
            "<li>Train models: <code>python src/model/train_model.py</code></li>"
            "</ul>"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Launch buttons
        full_app_btn = QPushButton("Launch Full Application (Slow)")
        full_app_btn.clicked.connect(self.launch_full_app)
        layout.addWidget(full_app_btn)
        
        test_btn = QPushButton("Run Functionality Test")
        test_btn.clicked.connect(self.run_test)
        layout.addWidget(test_btn)
        
        layout.addStretch()
    
    def _apply_branding(self) -> None:
        icon_path = self._resolve_brand_icon_path()
        if icon_path and icon_path.exists():
            app_icon = QIcon(str(icon_path))
            self.setWindowIcon(app_icon)
            QApplication.setWindowIcon(app_icon)

    def _resolve_brand_icon_path(self) -> Path:
        here = Path(__file__).resolve()
        dev_assets = here.parent / "assets" / "new-logos"
        brand_kit = here.parent.parent / "brand kit" / "new-logos"
        for base in (dev_assets, brand_kit):
            for name in ("PRIMARY-LOCKUP-cropped.svg", "PRIMARY-LOCKUP.svg", "type-logo.svg", "type-logo.png", "logo-no-background.png", "logo.png", "PRIMARY-LOCKUP-NO-BACKGROUND.png"):
                p = base / name
                if p.exists():
                    return p
        return dev_assets / "logo.png"
        
    def launch_full_app(self):
        """Launch the full application."""
        QMessageBox.information(
            self, 
            "Full Application", 
            "The full application will launch in a new window. This may take a few minutes to load models."
        )
        
        import subprocess
        subprocess.Popen([sys.executable, "src/app/main.py"])
        
    def run_test(self):
        """Run the functionality test."""
        import subprocess
        subprocess.Popen([sys.executable, "test_functionality.py"])


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Thoth Quick Launch")
    
    window = QuickLaunchWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
