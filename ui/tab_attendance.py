import tkinter as tk
from datetime import datetime
from tkinter import messagebox

import cv2

from attendance import AttendanceLogger
from config import (
    C_ACCENT, C_ACCENT2, C_BG, C_BORDER, C_PANEL, C_RED,
    C_SUCCESS, C_SURFACE, C_TEXT, C_TEXT_DIM, C_WARN,
    CAMERA_INDEX, CAMERA_TICK_MS, DEFAULT_CONF_THRESHOLD,
)
from face_engine import FaceDetector, LBPHRecognizer, UnknownFaceStore
from ui.widgets import (
    flat_button, horizontal_separator, make_camera_canvas,
    render_frame, scrolled_listbox, section_label,
    show_placeholder,
)


class AttendanceTab(tk.Frame):

    def __init__(self, parent, attendance: AttendanceLogger):
        super().__init__(parent, bg=C_BG)
        self._attendance   = attendance
        self._detector     = FaceDetector()
        self._recognizer   = LBPHRecognizer()
        self._unknown_store = UnknownFaceStore()
        self._running      = False
        self._cap          = None
        self._after_id     = None
        self._build()


    def _build(self):
        self._build_camera_panel()
        self._build_control_panel()

    def _build_camera_panel(self):
        left = tk.Frame(self, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                  padx=(12, 6), pady=12)
        self._canvas = make_camera_canvas(left)
        show_placeholder(
            self._canvas,
            "\t\t\t\t✅  Press  ▶ Start Attendance  to begin\nlive facial recognition.",
        )

    def _build_control_panel(self):
        right = tk.Frame(self, bg=C_PANEL, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 12), pady=12)
        right.pack_propagate(False)

        tk.Label(right, text="Attendance",
                 font=("Helvetica", 15, "bold"),
                 fg=C_ACCENT2, bg=C_PANEL).pack(
            pady=(22, 2), anchor="w", padx=18)

        tk.Label(right, text=datetime.now().strftime("%A, %d %B %Y"),
                 font=("Helvetica", 9), fg=C_TEXT_DIM, bg=C_PANEL).pack(
            anchor="w", padx=18, pady=(0, 10))

        self._status_var = tk.StringVar(value="●  Idle")
        self._status_lbl = tk.Label(
            right, textvariable=self._status_var,
            font=("Helvetica", 10, "bold"),
            fg=C_WARN, bg=C_PANEL,
        )
        self._status_lbl.pack(anchor="w", padx=18)

        horizontal_separator(right).pack(fill=tk.X, padx=18, pady=12)

        section_label(right, "Recognition Threshold (confidence)").pack(
            fill=tk.X, padx=18, pady=(0, 2))
        self._thresh_var = tk.IntVar(value=DEFAULT_CONF_THRESHOLD)
        self._thresh_display = tk.Label(
            right, text=str(DEFAULT_CONF_THRESHOLD),
            font=("Helvetica", 10, "bold"), fg=C_ACCENT, bg=C_PANEL)
        self._thresh_display.pack(anchor="e", padx=18)

        tk.Scale(
            right, from_=20, to=150, orient=tk.HORIZONTAL,
            variable=self._thresh_var,
            command=lambda v: self._thresh_display.config(text=str(int(float(v)))),
            bg=C_PANEL, fg=C_TEXT, troughcolor=C_SURFACE,
            highlightthickness=0, bd=0,
            activebackground=C_ACCENT, showvalue=False,
        ).pack(fill=tk.X, padx=18, pady=(0, 4))

        tk.Label(right, text="Lower value = stricter matching",
                 font=("Helvetica", 8), fg=C_TEXT_DIM, bg=C_PANEL).pack(
            anchor="w", padx=18, pady=(0, 12))

        self._start_btn = flat_button(
            right, "▶  Start Attendance",
            C_SUCCESS, fg=C_BG, cmd=self._on_start)
        self._start_btn.pack(fill=tk.X, padx=18, pady=4)

        self._stop_btn = flat_button(
            right, "■  Stop",
            C_SURFACE, fg=C_TEXT_DIM, cmd=self._on_stop)
        self._stop_btn.pack(fill=tk.X, padx=18, pady=4)
        self._stop_btn.config(state=tk.DISABLED)

        horizontal_separator(right).pack(fill=tk.X, padx=18, pady=12)

        stats = tk.Frame(right, bg=C_SURFACE)
        stats.pack(fill=tk.X, padx=18, pady=(0, 10))
        self._present_var = tk.StringVar(value="0")
        self._unknown_var = tk.StringVar(value="0")

        for label, var, color in [
            ("Present Today", self._present_var, C_SUCCESS),
            ("Unknown Faces", self._unknown_var, C_RED),
        ]:
            row = tk.Frame(stats, bg=C_SURFACE)
            row.pack(fill=tk.X, padx=10, pady=5)
            tk.Label(row, text=label, font=("Helvetica", 9),
                     fg=C_TEXT_DIM, bg=C_SURFACE).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var,
                     font=("Helvetica", 16, "bold"),
                     fg=color, bg=C_SURFACE).pack(side=tk.RIGHT)

        section_label(right, "Today's Log").pack(
            fill=tk.X, padx=18, pady=(8, 4))
        log_frame = tk.Frame(right, bg=C_SURFACE)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0, 14))
        self._log_list, _ = scrolled_listbox(log_frame)

        self._refresh_stats()


    def _refresh_stats(self):
        self._present_var.set(str(self._attendance.count()))
        self._unknown_var.set(str(self._unknown_store.count_today()))

    def _add_log_row(self, name: str, known: bool):
        t    = datetime.now().strftime("%H:%M:%S")
        icon = "✅" if known else "⚠"
        self._log_list.insert(0, f"{icon} {name:<22s} {t}")


    def _on_start(self):
        if not self._recognizer.load():
            messagebox.showerror(
                "Model Not Found",
                "No trained model was found.\n\n"
                "Please register faces and train the model first.",
            )
            return

        self._cap = cv2.VideoCapture(CAMERA_INDEX,cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam.")
            self._cap = None
            return

        self._running = True
        self._unknown_store.reset_cooldowns()
        self._start_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._status_var.set("●  Live")
        self._status_lbl.config(fg=C_SUCCESS)
        self._tick()

    def _on_stop(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None

        self._start_btn.config(state=tk.NORMAL)
        self._stop_btn.config(state=tk.DISABLED)
        self._status_var.set("●  Stopped")
        self._status_lbl.config(fg=C_WARN)
        show_placeholder(
            self._canvas,
            "Session ended.\nPress  ▶ Start Attendance  to begin again.",
        )


    def _tick(self):
        if not self._running or self._cap is None:
            return

        ret, frame = self._cap.read()
        if ret:
            self._process_frame(frame)

        self._after_id = self.after(CAMERA_TICK_MS, self._tick)

    def _process_frame(self, frame):
        gray, faces = self._detector.detect(frame)
        display     = frame.copy()
        threshold   = self._thresh_var.get()

        for (x, y, w, h) in faces:
            face_gray           = gray[y:y + h, x:x + w]
            name, conf, known   = self._recognizer.predict(face_gray, threshold)

            if known:
                color  = (80, 230, 120)
                if self._attendance.mark(name):
                    self.after(0, lambda n=name: self._add_log_row(n, True))
                    self.after(0, self._refresh_stats)
            else:
                color   = (60, 60, 240)
                grid_key = f"{x // 40}_{y // 40}"
                face_bgr = frame[y:y + h, x:x + w]
                if self._unknown_store.try_save(face_bgr, grid_key):
                    self.after(0, lambda: self._add_log_row("Unknown", False))
                    self.after(0, self._refresh_stats)

            self._draw_label(display, x, y, w, h, name, conf, color)

        self._draw_hud(display)
        render_frame(self._canvas, display)


    @staticmethod
    def _draw_label(display, x, y, w, h, name, conf, color):
        cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        conf_str  = f"{conf:.1f}" if conf < 999 else "—"
        tag_text  = f" {name}  [{conf_str}] "
        (tw, th), _ = cv2.getTextSize(
            tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display, (x, y - th - 14), (x + tw + 4, y), color, -1)
        cv2.putText(display, tag_text, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(self, display):
        h = display.shape[0]
        cv2.putText(
            display,
            f"Present: {self._attendance.count()}  |  "
            f"{datetime.now().strftime('%H:%M:%S')}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )