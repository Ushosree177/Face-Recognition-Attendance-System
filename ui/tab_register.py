import os
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

from config import (
    C_ACCENT, C_ACCENT2, C_BG, C_PANEL, C_SURFACE, C_TEXT_DIM,
    CAMERA_INDEX, CAMERA_TICK_MS, CAPTURE_MAX, CAPTURE_MIN,
    DEFAULT_CAPTURE_COUNT,
)
from face_engine import DatasetManager, FaceDetector
from ui.widgets import (
    flat_button, horizontal_separator, make_camera_canvas,
    render_frame, scrolled_listbox, section_label,
    show_placeholder, text_entry,
)


class RegisterTab(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent, bg=C_BG)
        self._detector   = FaceDetector()
        self._capturing  = False
        self._cap        = None
        self._after_id   = None
        self._count      = 0
        self._save_dir   = ""
        self._target     = DEFAULT_CAPTURE_COUNT
        self._build()

    def _build(self):
        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        left = tk.Frame(self, bg=C_PANEL, width=300)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 6), pady=12)
        left.pack_propagate(False)

        tk.Label(left, text="Register Face",
                 font=("Helvetica", 15, "bold"),
                 fg=C_ACCENT2, bg=C_PANEL).pack(
            pady=(22, 3), padx=18, anchor="w")
        tk.Label(left,
                 text="Capture images to build\nyour face dataset.",
                 font=("Helvetica", 9), fg=C_TEXT_DIM, bg=C_PANEL,
                 justify=tk.LEFT).pack(padx=18, anchor="w", pady=(0, 18))

        self._id_var   = tk.StringVar()
        self._name_var = tk.StringVar()

        for label, var in [("Employee / Student ID", self._id_var),
                            ("Full Name",             self._name_var)]:
            section_label(left, label).pack(fill=tk.X, padx=18, pady=(6, 2))
            text_entry(left, var).pack(fill=tk.X, padx=18, pady=(0, 4))

        section_label(left, "Images to Capture").pack(
            fill=tk.X, padx=18, pady=(10, 2))

        self._count_var = tk.IntVar(value=DEFAULT_CAPTURE_COUNT)
        self._count_display = tk.Label(
            left, text=str(DEFAULT_CAPTURE_COUNT),
            font=("Helvetica", 10, "bold"), fg=C_ACCENT, bg=C_PANEL)
        self._count_display.pack(anchor="e", padx=18)

        tk.Scale(
            left, from_=CAPTURE_MIN, to=CAPTURE_MAX,
            orient=tk.HORIZONTAL, variable=self._count_var,
            command=lambda v: self._count_display.config(text=str(int(float(v)))),
            bg=C_PANEL, fg=C_TEXT_DIM, troughcolor=C_SURFACE,
            highlightthickness=0, bd=0,
            activebackground=C_ACCENT, showvalue=False,
        ).pack(fill=tk.X, padx=18, pady=(0, 14))

        section_label(left, "Capture Progress").pack(
            fill=tk.X, padx=18, pady=(4, 3))
        self._prog = ttk.Progressbar(left, mode="determinate", length=240)
        self._prog.pack(padx=18, pady=(0, 3))
        self._prog_label = tk.Label(
            left, text="0 / 0 frames",
            font=("Helvetica", 8), fg=C_TEXT_DIM, bg=C_PANEL)
        self._prog_label.pack(anchor="w", padx=18)

        self._start_btn = flat_button(
            left, "▶  Start Capture", C_ACCENT, cmd=self._on_start)
        self._start_btn.pack(fill=tk.X, padx=18, pady=(16, 4))

        self._stop_btn = flat_button(
            left, "■  Stop Capture", C_SURFACE, fg=C_TEXT_DIM,
            cmd=self._on_stop)
        self._stop_btn.pack(fill=tk.X, padx=18, pady=4)
        self._stop_btn.config(state=tk.DISABLED)

        horizontal_separator(left).pack(fill=tk.X, padx=18, pady=14)
        section_label(left, "Registered Users").pack(
            fill=tk.X, padx=18, pady=(0, 5))

        list_frame = tk.Frame(left, bg=C_SURFACE)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 14))
        self._user_list, _ = scrolled_listbox(list_frame)
        self._refresh_users()

    def _build_right_panel(self):
        right = tk.Frame(self, bg=C_BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True,
                   padx=(6, 12), pady=12)
        self._canvas = make_camera_canvas(right)
        show_placeholder(
            self._canvas,
            "Camera feed will appear here\nafter clicking  ▶ Start Capture",
        )


    def _refresh_users(self):
        self._user_list.delete(0, tk.END)
        for u in DatasetManager.registered_users():
            self._user_list.insert(tk.END, f"  {u}")


    def _on_start(self):
        pid  = self._id_var.get().strip()
        name = self._name_var.get().strip()
        if not pid or not name:
            messagebox.showwarning(
                "Missing Info",
                "Please enter both an ID and a Name before capturing.",
            )
            return

        self._target   = self._count_var.get()
        self._save_dir = DatasetManager.user_folder(pid, name)
        self._count    = 0

        self._prog["maximum"] = self._target
        self._prog["value"]   = 0
        self._prog_label.config(text=f"0 / {self._target} frames")

        self._cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            self._cap = None
            return

        self._capturing = True
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._tick()

    def _tick(self):
        if not self._capturing or self._cap is None:
            return

        ret, frame = self._cap.read()
        if ret:
            gray, faces = self._detector.detect(frame)
            display = frame.copy()

            for (x, y, w, h) in faces:
                if self._count < self._target:
                    DatasetManager.save_face(
                        self._save_dir, self._count, gray[y:y + h, x:x + w]
                    )
                    self._count += 1
                    self._prog["value"] = self._count
                    self._prog_label.config(
                        text=f"{self._count} / {self._target} frames"
                    )

                pct   = int(self._count / self._target * 100)
                color = (80, 220, 130) if self._count < self._target else (80, 180, 255)
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, f"{pct}%", (x + 4, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.putText(
                display,
                f"Capturing: {self._count}/{self._target}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 200), 2,
            )
            render_frame(self._canvas, display)

            if self._count >= self._target:
                self._on_stop(done=True)
                return

        self._after_id = self.after(CAMERA_TICK_MS, self._tick)

    def _on_stop(self, done: bool = False):
        self._capturing = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None

        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._refresh_users()

        if done:
            messagebox.showinfo(
                "Capture Complete",
                f"Captured {self._count} images.\n\n"
                "Head to Train Model  to update the model.",
            )