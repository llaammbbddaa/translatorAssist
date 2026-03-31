import threading
import tkinter as tk
from tkinter import ttk

from translator_assist.audio import microphone_chunks
from translator_assist.transcribe import Transcriber
from translator_assist.translate import TranslatorBackend, BackendType


MIC_CHUNK_DURATION_SECONDS = 10.0 # adjusted from 5s to 10s, for more consistent translations


class TranslatorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Translator Assist")

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None

        # UI layout
        self._build_controls()
        self._build_output()

    def _build_controls(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        # Source language selector
        lang_frame = ttk.LabelFrame(controls, text="Source language")
        lang_frame.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.source_lang_var = tk.StringVar(value="en")
        ttk.Radiobutton(lang_frame, text="English", value="en", variable=self.source_lang_var).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Radiobutton(lang_frame, text="Spanish", value="es", variable=self.source_lang_var).grid(
            row=0, column=1, sticky="w"
        )

        # Buttons
        buttons_frame = ttk.Frame(controls)
        buttons_frame.grid(row=0, column=1, sticky="e")

        self.start_button = ttk.Button(buttons_frame, text="Start Mic", command=self.start_stream)
        self.start_button.grid(row=0, column=0, padx=(0, 5))

        self.stop_button = ttk.Button(buttons_frame, text="Stop Mic", command=self.stop_stream, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1)

        # Status label
        self.status_var = tk.StringVar(value="Idle")
        status_label = ttk.Label(controls, textvariable=self.status_var)
        status_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def _build_output(self) -> None:
        output_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        output_frame.grid(row=1, column=0, sticky="nsew")
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        label = ttk.Label(output_frame, text="Transcription | Translation")
        label.grid(row=0, column=0, sticky="w")

        self.text = tk.Text(output_frame, height=20, wrap="word")
        self.text.grid(row=1, column=0, sticky="nsew")
        output_frame.rowconfigure(1, weight=1)
        output_frame.columnconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.text["yscrollcommand"] = scrollbar.set

    def start_stream(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self.status_var.set("Listening…")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.text.insert(tk.END, "\n--- Started microphone streaming ---\n")
        self.text.see(tk.END)

        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_stream(self) -> None:
        self._stop_event.set()
        self.status_var.set("Stopping…")
        self.stop_button.config(state=tk.DISABLED)

    def _worker_loop(self) -> None:
        # Initialize models once per session
        source_lang = self.source_lang_var.get()
        target_lang = "es" if source_lang == "en" else "en"

        transcriber = Transcriber(model_name="small")
        # Use OPUS backend here because it currently gives the
        # most reliable EN↔ES translations in this setup.
        translator = TranslatorBackend(backend_type=BackendType.OPUS, source_lang=source_lang, target_lang=target_lang)

        try:
            for i, chunk in enumerate(microphone_chunks(chunk_duration=MIC_CHUNK_DURATION_SECONDS), start=1):
                if self._stop_event.is_set():
                    break

                transcription = transcriber.transcribe(chunk, source_lang=source_lang)
                translation = translator.translate(transcription)

                if not transcription and not translation:
                    continue

                # line = f"Chunk {i}:\n{transcription}  |  {translation}\n\n"
                line = f"{transcription}  |  {translation}\n\n"
                self._append_text(line)
        finally:
            self.root.after(0, self._on_stream_stopped)

    def _append_text(self, text: str) -> None:
        # Schedule UI update on the main thread
        def _do_insert() -> None:
            self.text.insert(tk.END, text)
            self.text.see(tk.END)

        self.root.after(0, _do_insert)

    def _on_stream_stopped(self) -> None:
        self.status_var.set("Idle")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.text.insert(tk.END, "--- Stopped microphone streaming ---\n")
        self.text.see(tk.END)


def main() -> None:
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
