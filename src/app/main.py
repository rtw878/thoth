#!/usr/bin/env python3
"""
Historia Scribe GUI Application

Main PyQt6 application for the Historia Scribe historical handwriting recognition tool.
Based on the blueprint in Section 8.2 of the roadmap.
"""

import sys
import yaml
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFileDialog, QComboBox,
    QProgressBar, QMessageBox, QSplitter, QMenuBar, QMenu, QStatusBar
)
from PyQt6.QtGui import QAction, QPixmap, QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from model.inference import ModelManager


class TranscriptionWorker(QThread):
    """Worker thread for running transcription to keep the UI responsive."""
    
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, image_path: Path, model_name: str, model_manager: ModelManager):
        super().__init__()
        self.image_path = image_path
        self.model_name = model_name
        self.model_manager = model_manager
    
    def run(self):
        """Run the transcription process."""
        try:
            self.progress.emit(25)
            
            # Check if model exists
            if self.model_name not in self.model_manager.list_models():
                self.error.emit(f"Model '{self.model_name}' not found")
                return
            
            self.progress.emit(50)
            
            # Perform actual transcription
            result = self.model_manager.transcribe_with_model(
                self.model_name, 
                self.image_path
            )
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(f"Transcription error: {str(e)}")


class MainWindow(QMainWindow):
    """Main application window for Historia Scribe."""
    
    def __init__(self):
        super().__init__()
        self.current_image_path: Optional[Path] = None
        self.current_model: Optional[str] = None
        self.model_manager: Optional[ModelManager] = None
        self.transcription_worker: Optional[TranscriptionWorker] = None
        
        self.init_ui()
        self.load_configuration()
        self.load_available_models()
    
    def init_ui(self) -> None:
        """Initialize the user interface."""
        self.setWindowTitle("Historia Scribe - Historical Handwriting Recognition")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Image display
        self.image_panel = self.create_image_panel()
        splitter.addWidget(self.image_panel)
        
        # Right panel - Controls and transcription
        self.control_panel = self.create_control_panel()
        splitter.addWidget(self.control_panel)
        
        # Set splitter proportions
        splitter.setSizes([700, 500])
        
        main_layout.addWidget(splitter)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_image_panel(self) -> QWidget:
        """Create the image display panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image label
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background: white;")
        self.image_label.setMinimumSize(400, 300)
        
        layout.addWidget(self.image_label)
        
        return panel
    
    def create_control_panel(self) -> QWidget:
        """Create the control and transcription panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Transcribe button
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.on_transcribe_clicked)
        self.transcribe_button.setEnabled(False)
        layout.addWidget(self.transcribe_button)
        
        # Transcription output
        layout.addWidget(QLabel("Transcription:"))
        self.transcription_text = QTextEdit()
        self.transcription_text.setPlaceholderText(
            "Transcription results will appear here..."
        )
        layout.addWidget(self.transcription_text)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.save_button = QPushButton("Save as Text")
        self.save_button.clicked.connect(self.on_save_text)
        self.save_button.setEnabled(False)
        export_layout.addWidget(self.save_button)
        
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.on_copy_text)
        self.copy_button.setEnabled(False)
        export_layout.addWidget(self.copy_button)
        
        layout.addLayout(export_layout)
        
        return panel
    
    def create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)
        
        # File menu
        file_menu = menu_bar.addMenu("File")
        
        open_image_action = QAction("Open Image...", self)
        open_image_action.setShortcut("Ctrl+O")
        open_image_action.triggered.connect(self.on_open_image)
        file_menu.addAction(open_image_action)
        
        open_folder_action = QAction("Open Folder...", self)
        open_folder_action.setShortcut("Ctrl+Shift+O")
        open_folder_action.triggered.connect(self.on_open_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.on_about)
        help_menu.addAction(about_action)
    
    def load_configuration(self) -> None:
        """Load application configuration."""
        config_path = Path("configs/config.yml")
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                self.model_manager = ModelManager(config)
                print("Configuration loaded successfully")
                
            except Exception as e:
                QMessageBox.warning(
                    self, 
                    "Configuration Error", 
                    f"Could not load configuration: {e}"
                )
                # Create default model manager
                self.model_manager = ModelManager({
                    'data_paths': {'models_dir': 'models'},
                    'app_settings': {'available_models': []}
                })
        else:
            QMessageBox.warning(
                self, 
                "Configuration Missing", 
                "Configuration file not found. Using default settings."
            )
            self.model_manager = ModelManager({
                'data_paths': {'models_dir': 'models'},
                'app_settings': {'available_models': []}
            })
    
    def load_available_models(self) -> None:
        """Load available model options."""
        if self.model_manager:
            models = self.model_manager.list_models()
            
            if models:
                self.model_combo.addItems(models)
                self.current_model = models[0]
                self.status_bar.showMessage(f"Loaded {len(models)} models")
            else:
                self.model_combo.addItem("No models available")
                self.transcribe_button.setEnabled(False)
                self.status_bar.showMessage("No models found. Please train models first.")
    
    def on_open_image(self) -> None:
        """Handle opening a single image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Historical Document Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tiff *.tif *.bmp)"
        )
        
        if file_path:
            self.load_image(Path(file_path))
    
    def on_open_folder(self) -> None:
        """Handle opening a folder of images."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Open Folder with Historical Document Images"
        )
        
        if folder_path:
            # TODO: Implement folder loading and batch processing
            QMessageBox.information(
                self,
                "Folder Loading",
                "Folder loading and batch processing will be implemented in a future version."
            )
    
    def load_image(self, image_path: Path) -> None:
        """Load and display an image."""
        try:
            pixmap = QPixmap(str(image_path))
            if pixmap.isNull():
                raise ValueError("Could not load image")
            
            # Scale image to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.current_image_path = image_path
            
            # Enable transcribe button if we have a model
            if self.current_model and self.current_model != "No models available":
                self.transcribe_button.setEnabled(True)
            
            self.status_bar.showMessage(f"Loaded: {image_path.name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load image: {e}")
    
    def on_model_changed(self, model_name: str) -> None:
        """Handle model selection change."""
        if model_name != "No models available":
            self.current_model = model_name
            self.status_bar.showMessage(f"Selected model: {model_name}")
            
            # Enable transcribe button if we have an image loaded
            if self.current_image_path:
                self.transcribe_button.setEnabled(True)
    
    def on_transcribe_clicked(self) -> None:
        """Handle transcription button click."""
        if not self.current_image_path or not self.current_model:
            QMessageBox.warning(self, "Warning", "Please load an image and select a model first.")
            return
        
        if not self.model_manager:
            QMessageBox.warning(self, "Warning", "Model manager not initialized.")
            return
        
        # Disable UI during transcription
        self.transcribe_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start transcription in worker thread
        self.transcription_worker = TranscriptionWorker(
            self.current_image_path, 
            self.current_model,
            self.model_manager
        )
        self.transcription_worker.progress.connect(self.on_transcription_progress)
        self.transcription_worker.finished.connect(self.on_transcription_finished)
        self.transcription_worker.error.connect(self.on_transcription_error)
        self.transcription_worker.start()
        
        self.status_bar.showMessage("Transcribing...")
    
    def on_transcription_progress(self, value: int) -> None:
        """Update progress bar during transcription."""
        self.progress_bar.setValue(value)
    
    def on_transcription_finished(self, result: str) -> None:
        """Handle completion of transcription."""
        self.transcription_text.setText(result)
        self.progress_bar.setVisible(False)
        self.transcribe_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.status_bar.showMessage("Transcription completed")
    
    def on_transcription_error(self, error_message: str) -> None:
        """Handle transcription errors."""
        self.progress_bar.setVisible(False)
        self.transcribe_button.setEnabled(True)
        QMessageBox.critical(self, "Transcription Error", error_message)
        self.status_bar.showMessage("Transcription failed")
    
    def on_save_text(self) -> None:
        """Save transcription to text file."""
        if not self.transcription_text.toPlainText():
            QMessageBox.warning(self, "Warning", "No transcription to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            "",
            "Text Files (*.txt)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.transcription_text.toPlainText())
                self.status_bar.showMessage(f"Transcription saved to {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {e}")
    
    def on_copy_text(self) -> None:
        """Copy transcription to clipboard."""
        if self.transcription_text.toPlainText():
            QApplication.clipboard().setText(self.transcription_text.toPlainText())
            self.status_bar.showMessage("Transcription copied to clipboard")
    
    def on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Historia Scribe",
            "Historia Scribe\n\n"
            "An AI-powered application for transcribing historical handwriting\n"
            "using state-of-the-art machine learning models.\n\n"
            "Version 0.1.0\n"
            "Built with PyQt6 and Hugging Face Transformers\n\n"
            "Features:\n"
            "• TrOCR-based handwriting recognition\n"
            "• LoRA fine-tuning for efficiency\n"
            "• Multi-model support\n"
            "• Cross-platform desktop application"
        )


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Historia Scribe")
    app.setApplicationVersion("0.1.0")
    
    # Set application style
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
