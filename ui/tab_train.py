import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from config import (
    C_ACCENT, C_ACCENT2, C_BG, C_PANEL, C_SURFACE,
    C_TEXT, C_TEXT_DIM, TRAINER_DIR, MODEL_FILE,
)
from face_engine import DatasetManager, LBPHTrainer
from ui.widgets import flat_button, log_append, scrolled_text


class TrainTab(tk.Frame):

    def __init__(self, parent, on_train_done: Callable = None):
        super().__init__(parent, bg=C_BG)
        self._trainer       = LBPHTrainer()
        self._training      = False
        self._on_train_done = on_train_done   # callback -> App.refresh_model_status
        self._build()

    def _build(self):
        card = tk.Frame(self, bg=C_PANEL)
        card.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=640, height=540)

        tk.Label(card, text="Train LBPH Model",
                 font=("Helvetica", 17, "bold"),
                 fg=C_ACCENT2, bg=C_PANEL).pack(pady=(28, 4))

        tk.Label(
            card,
            text=(
                "The LBPH (Local Binary Pattern Histogram) model will be\n"
                "trained on all registered face datasets."
            ),
            font=("Helvetica", 9), fg=C_TEXT_DIM, bg=C_PANEL,
            justify=tk.CENTER,
        ).pack(pady=(0, 18))

        info_bg = tk.Frame(card, bg=C_SURFACE)
        info_bg.pack(fill=tk.X, padx=40, pady=(0, 14))
        self._info_label = tk.Label(
            info_bg, text="Scanning...",
            font=("Helvetica", 10), fg=C_TEXT, bg=C_SURFACE,
            anchor="w", padx=12, pady=8,
        )
        self._info_label.pack(fill=tk.X)

        # Fixed height so progressbar + button are never clipped outside the card
        log_outer = tk.Frame(card, bg=C_SURFACE, height=220)
        log_outer.pack(fill=tk.X, expand=False, padx=40, pady=(0, 10))
        log_outer.pack_propagate(False)
        self._log = scrolled_text(log_outer)

        self._prog = ttk.Progressbar(card, mode="indeterminate", length=560)
        self._prog.pack(pady=(6, 10))

        self._train_btn = flat_button(
            card, "Train Model Now", C_ACCENT,
            cmd=self._on_train, font_size=12, pady=11,
        )
        self._train_btn.pack(pady=(0, 24))

        self._refresh_summary()

    def _refresh_summary(self):
        summary      = DatasetManager.dataset_summary()
        model_exists = os.path.exists(os.path.join(TRAINER_DIR, MODEL_FILE))
        status       = "Model trained" if model_exists else "Model not yet trained"
        self._info_label.config(
            text=(
                f"  Registered users: {summary['users']}   |   "
                f"Total images: {summary['images']}   |   {status}"
            )
        )

    def _on_train(self):
        if self._training:
            return

        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

        self._prog.start(12)
        self._train_btn.config(state=tk.DISABLED, text="  Training...  ")
        self._training = True
        self._refresh_summary()
        log_append(self._log, "Starting LBPH training...\n", "info")

        def _progress(folder, done, total):
            self.after(0, lambda: log_append(
                self._log,
                f"  [{folder}]   {done}/{total} images processed",
                "folder",
            ))

        def _done(success: bool, msg: str):
            def _update_ui():
                self._prog.stop()
                self._train_btn.config(state=tk.NORMAL, text="Train Model Now")
                self._training = False
                self._refresh_summary()   # updates "Model trained" in the info bar

                log_append(
                    self._log,
                    f"\n{'SUCCESS' if success else 'FAILED'}: {msg}",
                    "ok" if success else "err",
                )

                # Tell App to refresh the header label in real time
                if success and self._on_train_done:
                    self._on_train_done()

                if success:
                    messagebox.showinfo("Training Complete", f"  {msg}")
                else:
                    messagebox.showerror("Training Failed", msg)

            self.after(0, _update_ui)

        threading.Thread(
            target=self._trainer.train,
            args=(_progress, _done),
            daemon=True,
        ).start()