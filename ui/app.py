import tkinter as tk
from tkinter import ttk

from attendance import AttendanceLogger
from config import (
    ATTENDANCE_DIR, C_BG, C_PANEL, C_SUCCESS, C_TEXT_DIM, C_WARN,
    DATASET_DIR, UNKNOWN_DIR,
)
from face_engine import LBPHRecognizer
from ui.tab_attendance import AttendanceTab
from ui.tab_register import RegisterTab
from ui.tab_train import TrainTab
from ui.widgets import apply_styles


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self._ensure_dirs()

        self._attendance = AttendanceLogger()
        self._recognizer = LBPHRecognizer()   # used only to probe model status

        self.title("FaceAttend - Facial Recognition Attendance System")
        self.geometry("1120x700")
        self.minsize(900, 580)
        self.configure(bg=C_BG)

        apply_styles(self)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)


    @staticmethod
    def _ensure_dirs():
        import os
        from config import DATASET_DIR, TRAINER_DIR, UNKNOWN_DIR, ATTENDANCE_DIR
        for d in [DATASET_DIR, TRAINER_DIR, UNKNOWN_DIR, ATTENDANCE_DIR]:
            os.makedirs(d, exist_ok=True)

    def refresh_model_status(self):
        """Re-probe the model file and update the header label live."""
        ready = self._recognizer.load()
        self._model_status_lbl.config(
            text="Model: Ready \u2713" if ready else "Model: Not trained \u26a0",
            fg=C_SUCCESS if ready else C_WARN,
        )


    def _build(self):
        self._build_header()
        self._build_notebook()
        self._build_status_bar()

    def _build_header(self):
        hdr = tk.Frame(self, bg=C_PANEL, height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="\u2b21  FaceAttend",
                 font=("Helvetica", 19, "bold"),
                 fg="#00e5c3", bg=C_PANEL).pack(side=tk.LEFT, padx=22, pady=10)

        tk.Label(hdr, text="LBPH Facial Recognition Attendance System",
                 font=("Helvetica", 9), fg=C_TEXT_DIM, bg=C_PANEL).pack(
            side=tk.LEFT, pady=10)

        # Store the label so refresh_model_status() can update it at any time
        self._model_status_lbl = tk.Label(
            hdr, text="",
            font=("Helvetica", 9, "bold"),
            bg=C_PANEL,
        )
        self._model_status_lbl.pack(side=tk.RIGHT, padx=22)
        self.refresh_model_status()   # set correct text/colour on startup

    def _build_notebook(self):
        nb = ttk.Notebook(self, style="FA.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True)

        nb.add(RegisterTab(nb),
               text="    Register Face  ")
        # Pass refresh_model_status as a callback so TrainTab triggers it after training
        nb.add(TrainTab(nb, on_train_done=self.refresh_model_status),
               text="    Train Model  ")
        nb.add(AttendanceTab(nb, self._attendance),
               text="Mark Attendance  ")

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=C_PANEL, height=24)
        bar.pack(fill=tk.X, side=tk.BOTTOM)


        bar_frame = tk.Frame(bar, bg=C_PANEL)
        bar_frame.pack(fill=tk.X, padx=6)

        left_label = tk.Label(
            bar_frame,
            text=(
                f"Dataset: {DATASET_DIR}   |   "
                f"Attendance: {ATTENDANCE_DIR}   |   "
                f"Unknown faces: {UNKNOWN_DIR}"
            ),
            font=("Helvetica", 8),
            fg=C_TEXT_DIM,
            bg=C_PANEL,
            anchor="w",
        )
        left_label.pack(side="left")

        right_label = tk.Label(
            bar_frame,
            text="Project by Yazhini & Ushosree",
            font=("Helvetica", 8),
            fg=C_TEXT_DIM,
            bg=C_PANEL,
            anchor="e",
        )
        right_label.pack(side="right")

    def _on_close(self):
        self.destroy()