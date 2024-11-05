import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QVBoxLayout, QWidget, QLineEdit,
                             QTextEdit, QMessageBox, QHBoxLayout, QFrame, QToolButton,
                             QListWidget, QListWidgetItem, QInputDialog, QDialog,
                             QCheckBox, QDialogButtonBox, QProgressDialog, QFormLayout)
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QSettings

class ResponseThread(QThread):
    response_generated = pyqtSignal(str)

    def __init__(self, model, tokenizer, conversation_history):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.conversation_history = conversation_history

    def run(self):
        try:
            import torch

            # Überprüfen, ob eine GPU verfügbar ist
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

            with torch.no_grad():
                # Eingabe vorbereiten
                inputs = self.tokenizer(
                    self.conversation_history,
                    return_tensors="pt",
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=1024  # Maximale Länge erhöhen, um die Gesprächshistorie zu berücksichtigen
                ).to(device)

                # Antwort generieren mit optimierten Decodierungsparametern
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    repetition_penalty=1.2,
                )

                # Ausgabe dekodieren
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Signal mit der generierten Antwort senden
                self.response_generated.emit(response)

        except Exception as e:
            error_message = f"Fehler bei der Generierung der Antwort: {e}"
            self.response_generated.emit(error_message)

class ModelLoaderThread(QThread):
    model_loaded = pyqtSignal(object, object)  # Signal mit Modell und Tokenizer oder Fehler

    def __init__(self, model_name, requires_token, HF_TOKEN=None):
        super().__init__()
        self.model_name = model_name
        self.requires_token = requires_token
        self.HF_TOKEN = HF_TOKEN

    def run(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.requires_token:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.HF_TOKEN)
                model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=self.HF_TOKEN)
            else:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Prüfen und setzen des pad_token, falls notwendig
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Signal senden, wenn das Modell geladen ist
            self.model_loaded.emit(model, tokenizer)
        except Exception as e:
            # Fehler an das Hauptfenster weitergeben
            self.model_loaded.emit(None, str(e))

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.layout = QVBoxLayout()

        # Checkbox für den Dunkelmodus
        self.dark_mode_checkbox = QCheckBox("Dunkelmodus aktivieren")
        self.dark_mode_checkbox.setChecked(parent.is_dark_mode)
        self.layout.addWidget(self.dark_mode_checkbox)

        # Buttons für OK und Abbrechen
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

    def accept(self):
        # Einstellungen speichern
        parent = self.parent()
        parent.is_dark_mode = self.dark_mode_checkbox.isChecked()
        parent.apply_style()
        super().accept()

class AddModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modell hinzufügen")
        self.layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.display_name_input = QLineEdit()
        self.model_name_input = QLineEdit()
        self.requires_token_checkbox = QCheckBox("Benötigt Hugging Face Token")
        form_layout.addRow("Anzeigename:", self.display_name_input)
        form_layout.addRow("Modellname:", self.model_name_input)
        form_layout.addRow("", self.requires_token_checkbox)

        self.layout.addLayout(form_layout)

        # Buttons für OK und Abbrechen
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.validate_and_accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

    def validate_and_accept(self):
        if not self.display_name_input.text() or not self.model_name_input.text():
            QMessageBox.warning(self, "Eingabefehler", "Bitte füllen Sie alle Felder aus.")
            return
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Chat Application")
        self.setGeometry(100, 100, 1200, 800)

        # Einstellungen laden
        self.settings = QSettings("IhrUnternehmen", "LLMChatApplication")
        self.is_dark_mode = self.settings.value("is_dark_mode", False, type=bool)

        # Initialisierung des Chat-Modells
        self.chat_model = None
        self.tokenizer = None
        self.HF_TOKEN = None  # Initialisierung des Tokens

        # Sammlungen für LocalDocs
        self.collections = []

        # Verfügbare Modelle
        self.available_models = [
            {"name": "dbmdz/german-gpt2", "model_name": "dbmdz/german-gpt2", "requires_token": False},
            {"name": "deepset/gbert-base", "model_name": "deepset/gbert-base", "requires_token": False},
            {"name": "bert-base-german-cased", "model_name": "bert-base-german-cased", "requires_token": False},
            {"name": "microsoft/DialoGPT-medium", "model_name": "microsoft/DialoGPT-medium", "requires_token": False},
            # Weitere Modelle können hier hinzugefügt werden
        ]

        self.selected_model = None  # Initialisieren mit None

        # Gesprächshistorie
        self.conversation_history = ""

        # GUI-Setup
        self.setup_ui()

        # Stylesheet anwenden
        self.apply_style()

        # Standardmäßig das Chat-Interface laden
        self.load_chats()

    def setup_ui(self):
        # Hauptlayout
        main_layout = QHBoxLayout()

        # Seitenleiste erstellen
        sidebar = QFrame()
        sidebar.setFixedWidth(100)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setAlignment(Qt.AlignTop)

        # Stylesheet für die Seitenleiste
        sidebar.setStyleSheet("background-color: #2673a3;")

        # Buttons mit Icons erstellen
        self.button_chats = QToolButton()
        self.button_chats.setIcon(QIcon('icons/chat.png'))
        self.button_chats.setIconSize(QSize(48, 48))
        self.button_chats.setToolTip("Chats")
        self.button_chats.clicked.connect(self.load_chats)
        sidebar_layout.addWidget(self.button_chats)

        self.button_models = QToolButton()
        self.button_models.setIcon(QIcon('icons/model.png'))
        self.button_models.setIconSize(QSize(48, 48))
        self.button_models.setToolTip("Models")
        self.button_models.clicked.connect(self.load_models)
        sidebar_layout.addWidget(self.button_models)

        self.button_localdocs = QToolButton()
        self.button_localdocs.setIcon(QIcon('icons/localdocs.png'))
        self.button_localdocs.setIconSize(QSize(48, 48))
        self.button_localdocs.setToolTip("LocalDocs")
        self.button_localdocs.clicked.connect(self.load_localdocs)
        sidebar_layout.addWidget(self.button_localdocs)

        # Einstellungen-Button
        self.button_settings = QToolButton()
        self.button_settings.setIcon(QIcon('icons/settings.png'))
        self.button_settings.setIconSize(QSize(48, 48))
        self.button_settings.setToolTip("Einstellungen")
        self.button_settings.clicked.connect(self.open_settings)
        sidebar_layout.addWidget(self.button_settings)

        sidebar.setLayout(sidebar_layout)

        # Hauptinhalt
        self.content = QWidget()
        self.content_layout = QVBoxLayout()

        # Top-Bereich für dauerhafte Widgets
        self.top_content_layout = QVBoxLayout()
        self.main_content_layout = QVBoxLayout()

        self.content_layout.addLayout(self.top_content_layout)
        self.content_layout.addLayout(self.main_content_layout)
        self.content.setLayout(self.content_layout)

        # Label für das aktuelle Modell
        self.model_label = QLabel()
        self.model_label.setStyleSheet("font-size: 14px; color: #01565c;")
        self.top_content_layout.addWidget(self.model_label)
        self.update_model_label()  # Label initialisieren

        # Layouts zusammenführen
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.content)

        # Hauptwidget
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def apply_style(self):
        if self.is_dark_mode:
            # Dunkelmodus Stylesheet
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #405468;
                }
                QLabel {
                    color: #ECF0F1;
                }
                QPushButton {
                    background-color: #3f80ab;
                    color: white;
                    font-size: 14px;
                    padding: 8px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #405468;
                }
                QLineEdit, QTextEdit {
                    background-color: #34495E;
                    color: #ECF0F1;
                    border: 1px solid #BDC3C7;
                    padding: 6px;
                    border-radius: 5px;
                    font-size: 14px;
                }
                QListWidget {
                    background-color: #34495E;
                    color: #ECF0F1;
                    border: 1px solid #BDC3C7;
                    border-radius: 5px;
                    font-size: 14px;
                }
                QCheckBox {
                    color: #ECF0F1;
                    font-size: 14px;
                }
            """)
        else:
            # Hellmodus Stylesheet
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #ECF0F1;
                }
                QLabel {
                    color: #34495E;
                }
                QPushButton {
                    background-color: #3498DB;
                    color: white;
                    font-size: 14px;
                    padding: 8px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #2980B9;
                }
                QLineEdit, QTextEdit {
                    border: 1px solid #BDC3C7;
                    padding: 6px;
                    border-radius: 5px;
                    font-size: 14px;
                }
                QListWidget {
                    border: 1px solid #BDC3C7;
                    border-radius: 5px;
                    font-size: 14px;
                }
                QCheckBox {
                    font-size: 14px;
                }
            """)
        # Einstellungen speichern
        self.settings.setValue("is_dark_mode", self.is_dark_mode)

    def open_settings(self):
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec_()

    def clear_content(self):
        """Entfernt alle Widgets und Layouts aus dem Hauptinhaltsbereich, aber nicht aus dem Top-Bereich."""
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                elif item.layout() is not None:
                    clear_layout(item.layout())
                else:
                    pass  # Es könnte sich um einen Spacer handeln
        clear_layout(self.main_content_layout)

    def load_chats(self):
        self.clear_content()
        self.label = QLabel("Chat-Funktion geladen.")
        self.label.setStyleSheet("font-size: 16px;")
        self.main_content_layout.addWidget(self.label)

        # Textfeld für den Chat-Verlauf
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.main_content_layout.addWidget(self.chat_display)

        # Eingabefeld für den Chat
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Nachricht eingeben...")
        self.main_content_layout.addWidget(self.chat_input)

        # Verbindung zum Senden per Enter-Taste
        self.chat_input.returnPressed.connect(self.handle_chat)

        # Checkbox für die Verwendung von LocalDocs
        self.use_localdocs_checkbox = QCheckBox("LocalDocs verwenden")
        self.use_localdocs_checkbox.setChecked(False)  # Standardmäßig deaktiviert
        self.main_content_layout.addWidget(self.use_localdocs_checkbox)

        # Senden-Button
        self.button_send = QPushButton("Senden")
        self.button_send.clicked.connect(self.handle_chat)
        self.main_content_layout.addWidget(self.button_send)

        # Hinweis, wenn Modell nicht geladen ist
        if self.chat_model is None:
            self.chat_display.append("<i>Kein Modell geladen. Bitte wählen Sie ein Modell unter 'Models' aus und laden Sie es.</i>")

    def update_model_label(self):
        """Aktualisiert das Label mit dem Namen des aktuell geladenen Modells."""
        if self.chat_model is not None:
            self.model_label.setText(f"Aktuelles Modell: {self.selected_model['name']}")
        else:
            self.model_label.setText("Kein Modell geladen.")

    def load_chat_model(self):
        if self.selected_model is None:
            QMessageBox.warning(self, "Keine Auswahl", "Bitte wählen Sie ein Modell aus der Liste aus.")
            return

        if self.chat_model is None or self.tokenizer is None:
            # Anzeigen des Fortschrittsdialogs
            self.progress_dialog = QProgressDialog("Modell wird geladen...", None, 0, 0, self)
            self.progress_dialog.setWindowTitle("Bitte warten")
            self.progress_dialog.setWindowModality(Qt.ApplicationModal)
            self.progress_dialog.setCancelButton(None)
            self.progress_dialog.setRange(0, 0)  # Unbestimmter Fortschrittsbalken
            self.progress_dialog.show()

            # Thread zum Laden des Modells starten
            self.model_loader_thread = ModelLoaderThread(
                self.selected_model['model_name'],
                self.selected_model.get('requires_token', False),
                self.HF_TOKEN
            )
            self.model_loader_thread.model_loaded.connect(self.on_model_loaded)
            self.model_loader_thread.start()

    def on_model_loaded(self, model, tokenizer_or_error):
        # Fortschrittsdialog schließen
        self.progress_dialog.close()

        if model is not None:
            # Modell erfolgreich geladen
            self.chat_model = model
            self.tokenizer = tokenizer_or_error  # Hier ist es der Tokenizer

            self.label.setText(f"Modell {self.selected_model['name']} geladen.")
            self.update_model_label()  # Label aktualisieren
        else:
            # Fehler beim Laden
            error_message = tokenizer_or_error  # Hier ist es der Fehlertext
            QMessageBox.critical(self, "Fehler", f"Laden des Modells fehlgeschlagen: {error_message}")

    def load_models(self):
        self.clear_content()
        self.label = QLabel("Wähle ein Modell aus der Liste oder füge ein neues hinzu:")
        self.label.setStyleSheet("font-size: 16px;")
        self.main_content_layout.addWidget(self.label)

        self.model_list = QListWidget()
        self.update_model_list()
        self.model_list.currentRowChanged.connect(self.change_model)
        self.main_content_layout.addWidget(self.model_list)

        # Buttons zum Laden und Hinzufügen von Modellen
        button_layout = QHBoxLayout()

        self.load_model_button = QPushButton("Load Model")
        self.load_model_button.clicked.connect(self.load_selected_model)
        button_layout.addWidget(self.load_model_button)

        self.add_model_button = QPushButton("Add Model")
        self.add_model_button.clicked.connect(self.add_model)
        button_layout.addWidget(self.add_model_button)

        self.main_content_layout.addLayout(button_layout)

    def update_model_list(self):
        self.model_list.clear()
        for model in self.available_models:
            item = QListWidgetItem(model['name'])
            item.setData(Qt.UserRole, model)
            self.model_list.addItem(item)

        # Zeige das aktuell ausgewählte Modell an
        if self.selected_model:
            try:
                current_index = self.available_models.index(self.selected_model)
                self.model_list.setCurrentRow(current_index)
            except ValueError:
                pass  # Modell nicht gefunden

    def change_model(self, index):
        if 0 <= index < len(self.available_models):
            self.selected_model = self.available_models[index]
            # Aktualisiere das Label
            self.label.setText(f"Ausgewähltes Modell: {self.selected_model['name']}")

    def load_selected_model(self):
        if self.selected_model is None:
            QMessageBox.warning(self, "Keine Auswahl", "Bitte wählen Sie ein Modell aus der Liste aus.")
            return

        requires_token = self.selected_model.get('requires_token', False)
        if requires_token:
            # Eingabefeld für den Token
            token, ok = QInputDialog.getText(self, "Hugging Face Token", "Bitte geben Sie Ihren Hugging Face Access Token ein:")
            if ok and token:
                self.HF_TOKEN = token
            else:
                QMessageBox.warning(self, "Token benötigt", "Das ausgewählte Modell erfordert einen Hugging Face Access Token.")
                return
        else:
            self.HF_TOKEN = None

        # Modell neu laden
        self.chat_model = None
        self.tokenizer = None
        self.conversation_history = ""  # Gesprächshistorie zurücksetzen
        self.load_chat_model()

    def add_model(self):
        add_model_dialog = AddModelDialog(self)
        if add_model_dialog.exec_() == QDialog.Accepted:
            display_name = add_model_dialog.display_name_input.text()
            model_name = add_model_dialog.model_name_input.text()
            requires_token = add_model_dialog.requires_token_checkbox.isChecked()

            # Neues Modell zur Liste hinzufügen
            new_model = {
                "name": display_name,
                "model_name": model_name,
                "requires_token": requires_token
            }
            self.available_models.append(new_model)
            self.update_model_list()
            QMessageBox.information(self, "Modell hinzugefügt", f"Das Modell '{display_name}' wurde hinzugefügt.")

    def load_localdocs(self):
        self.clear_content()
        self.label = QLabel("Verwalte deine Dokumentensammlungen:")
        self.label.setStyleSheet("font-size: 16px;")
        self.main_content_layout.addWidget(self.label)

        # Liste der Sammlungen
        self.collection_list = QListWidget()
        self.update_collection_list()
        self.main_content_layout.addWidget(self.collection_list)

        # Buttons zum Hinzufügen und Entfernen von Sammlungen
        button_layout = QHBoxLayout()

        self.button_add_collection = QPushButton("Add Collection")
        self.button_add_collection.clicked.connect(self.add_collection)
        button_layout.addWidget(self.button_add_collection)

        self.button_remove_collection = QPushButton("Remove Collection")
        self.button_remove_collection.clicked.connect(self.remove_collection)
        button_layout.addWidget(self.button_remove_collection)

        self.main_content_layout.addLayout(button_layout)

    def update_collection_list(self):
        self.collection_list.clear()
        for collection in self.collections:
            item = QListWidgetItem(f"{collection['name']} - {len(collection['filenames'])} Dateien")
            item.setData(Qt.UserRole, collection)
            self.collection_list.addItem(item)

    def add_collection(self):
        # Name für die Sammlung abfragen
        collection_name, ok = QInputDialog.getText(self, "Sammlung benennen", "Name der Sammlung:")
        if ok and collection_name:
            # Ordner auswählen
            folder = QFileDialog.getExistingDirectory(self, "Ordner auswählen")
            if folder:
                self.label.setText(f"PDFs aus {folder} werden indexiert.")
                QApplication.processEvents()
                import os
                from PyPDF2 import PdfReader
                from sentence_transformers import SentenceTransformer
                import faiss
                import numpy as np

                # Initialisiere das Modell
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                texts = []
                filenames = []

                # Text aus PDFs extrahieren
                for filename in os.listdir(folder):
                    if filename.endswith('.pdf'):
                        pdf_path = os.path.join(folder, filename)
                        try:
                            reader = PdfReader(pdf_path)
                            text = ''
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text
                            if text:
                                texts.append(text)
                                filenames.append(filename)
                        except Exception as e:
                            print(f"Fehler beim Lesen von {filename}: {e}")

                # Prüfen, ob Texte extrahiert wurden
                if not texts:
                    self.label.setText("Keine Texte in den PDFs gefunden.")
                    return

                # Embeddings erstellen
                embeddings = model.encode(texts, convert_to_numpy=True)

                # Embeddings mit Faiss indexieren
                dimension = embeddings.shape[1]
                index = faiss.IndexFlatL2(dimension)
                index.add(embeddings)

                # Sammlung speichern
                collection = {
                    "name": collection_name,
                    "folder": folder,
                    "filenames": filenames,
                    "texts": texts,  # Texte hinzufügen
                    "index": index,
                    "model": model
                }
                self.collections.append(collection)
                self.update_collection_list()

                self.label.setText(f"Sammlung '{collection_name}' hinzugefügt.")

    def remove_collection(self):
        selected_items = self.collection_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Keine Auswahl", "Bitte wählen Sie eine Sammlung zum Entfernen aus.")
            return
        for item in selected_items:
            collection = item.data(Qt.UserRole)
            self.collections.remove(collection)
        self.update_collection_list()
        self.label.setText("Sammlung(en) entfernt.")

    def handle_chat(self):
        if self.chat_model is None or self.tokenizer is None:
            QMessageBox.warning(self, "Kein Modell geladen", "Bitte laden Sie zuerst ein Modell unter 'Models'.")
            return

        user_input = self.chat_input.text()
        if user_input:
            self.chat_display.append(f"<b>Sie:</b> {user_input}")
            self.chat_input.clear()

            # Append user input to conversation history
            self.conversation_history += f"Sie: {user_input}\n"

            # Überprüfen, ob LocalDocs verwendet werden sollen
            if self.use_localdocs_checkbox.isChecked():
                # Kontext aus LocalDocs abrufen
                localdocs_context = self.get_localdocs_context(user_input)
                if localdocs_context:
                    # Kontext dem Gesprächsverlauf hinzufügen
                    self.conversation_history += f"{localdocs_context}\n"

            # Statusmeldung anzeigen (Nachricht beibehalten)
            self.chat_display.append(f"<i>{self.selected_model['name']} generiert eine Antwort...</i>")

            # Antwort vom Modell in einem separaten Thread generieren
            self.response_thread = ResponseThread(self.chat_model, self.tokenizer, self.conversation_history)
            self.response_thread.response_generated.connect(self.display_response)
            self.response_thread.start()

    def get_localdocs_context(self, query):
        # Überprüfen, ob Sammlungen vorhanden sind
        if not self.collections:
            QMessageBox.warning(self, "Keine Sammlungen", "Bitte fügen Sie zuerst eine Sammlung unter 'LocalDocs' hinzu.")
            return ""

        # Kontext sammeln
        context = ""
        for collection in self.collections:
            index = collection['index']
            model = collection['model']
            texts = collection['texts']  # Texte aus der Sammlung

            # Anfrage einbetten
            query_embedding = model.encode([query], convert_to_numpy=True)

            # Index durchsuchen
            k = 3  # Anzahl der abzurufenden relevanten Dokumente
            distances, indices = index.search(query_embedding, k)

            # Relevante Texte sammeln
            for idx in indices[0]:
                if idx < len(texts):
                    context += texts[idx] + "\n\n"

        # Kontext begrenzen, um zu lange Eingaben zu vermeiden
        max_context_length = 1000  # Maximal 1000 Zeichen im Kontext
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."

        return context.strip()

    def display_response(self, response):
        # Generierte Antwort im Chat-Fenster anzeigen
        self.chat_display.append(f"<b>{self.selected_model['name']}:</b> {response}")
        
        # Append response to conversation history
        self.conversation_history += f"{self.selected_model['name']}: {response}\n"

    def handle_user_query_button(self):
        query = self.query_input.text()
        if query:
            self.handle_user_query(query)

    def handle_user_query(self, query):
        # Auswahl der Sammlung
        if not self.collections:
            QMessageBox.warning(self, "Keine Sammlungen", "Bitte fügen Sie zuerst eine Sammlung hinzu.")
            return

        collection_names = [c['name'] for c in self.collections]
        collection_name, ok = QInputDialog.getItem(self, "Sammlung auswählen", "Sammlung:", collection_names, 0, False)
        if ok and collection_name:
            collection = next((c for c in self.collections if c['name'] == collection_name), None)
            if collection:
                index = collection['index']
                model = collection['model']
                filenames = collection['filenames']

                # Anfrage verarbeiten
                query_embedding = model.encode([query], convert_to_numpy=True)

                # Index durchsuchen
                k = 5  # Anzahl der nächsten Nachbarn
                distances, indices = index.search(query_embedding, k)

                # Relevanteste Dokumente erhalten
                results = []
                for idx in indices[0]:
                    if idx < len(filenames):
                        results.append(filenames[idx])

                if results:
                    self.label.setText(f"Relevante Dokumente: {', '.join(results)}")
                else:
                    self.label.setText("Keine relevanten Dokumente gefunden.")
            else:
                self.label.setText("Sammlung nicht gefunden.")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Icons laden
    if not os.path.exists('icons'):
        os.makedirs('icons')

    # Platzhalter-Icons erstellen, falls sie nicht existieren
    def create_placeholder_icon(path, color):
        if not os.path.exists(path):
            from PIL import Image, ImageDraw
            img = Image.new('RGBA', (64, 64), color)
            img.save(path)

    create_placeholder_icon('icons/chat.png', (52, 152, 219, 255))       # Blau
    create_placeholder_icon('icons/model.png', (46, 204, 113, 255))     # Grün
    create_placeholder_icon('icons/localdocs.png', (155, 89, 182, 255)) # Lila
    create_placeholder_icon('icons/settings.png', (241, 196, 15, 255))  # Gelb

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

