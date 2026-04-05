import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from config import (
    C_ACCENT, C_BG, C_BORDER, C_PANEL, C_SUCCESS,
    C_SURFACE, C_TEXT, C_TEXT_DIM,
)


def apply_styles(root: tk.Tk):
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(
        "FA.TNotebook",
        background=C_BG,
        borderwidth=0,
        tabmargins=0,
    )
    style.configure(
        "FA.TNotebook.Tab",
        background=C_PANEL,
        foreground=C_TEXT_DIM,
        padding=[22, 9],
        font=("Helvetica", 10, "bold"),
        borderwidth=0,
    )
    style.map(
        "FA.TNotebook.Tab",
        background=[("selected", C_SURFACE), ("active", C_SURFACE)],
        foreground=[("selected", "#00e5c3"), ("active", C_TEXT)],
    )
    style.configure(
        "TProgressbar",
        troughcolor=C_SURFACE,
        background=C_ACCENT,
        borderwidth=0,
        thickness=6,
    )


def flat_button(parent, text: str, bg: str = C_ACCENT, fg: str = C_TEXT, cmd=None, font_size: int = 10, pady: int = 9,) -> tk.Button:
    return tk.Button(
        parent,
        text=text,
        command=cmd,
        bg=bg,
        fg=fg,
        activebackground=bg,
        activeforeground=fg,
        font=("Helvetica", font_size, "bold"),
        relief=tk.FLAT,
        bd=0,
        padx=14,
        pady=pady,
        cursor="hand2",
        highlightthickness=0,
    )


def section_label(parent, text: str) -> tk.Label:
    return tk.Label(
        parent,
        text=text,
        font=("Helvetica", 9, "bold"),
        fg=C_TEXT_DIM,
        bg=C_PANEL,
        anchor="w",
    )


def text_entry(parent, textvariable: tk.StringVar) -> tk.Entry:
    return tk.Entry(
        parent,
        textvariable=textvariable,
        font=("Helvetica", 11),
        bg=C_SURFACE,
        fg=C_TEXT,
        insertbackground=C_TEXT,
        relief=tk.FLAT,
        bd=6,
        highlightthickness=1,
        highlightcolor=C_ACCENT,
        highlightbackground=C_BORDER,
    )


def horizontal_separator(parent) -> tk.Frame:
    return tk.Frame(parent, bg=C_BORDER, height=1)


def scrolled_listbox(parent, **kwargs) -> tuple[tk.Listbox, tk.Scrollbar]:
    sb = tk.Scrollbar(parent, bg=C_SURFACE)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    lb = tk.Listbox(
        parent,
        bg="#090910",
        fg=C_TEXT,
        font=("Courier", 8),
        relief=tk.FLAT,
        bd=0,
        selectbackground=C_ACCENT,
        yscrollcommand=sb.set,
        **kwargs,
    )
    lb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    sb.config(command=lb.yview)
    return lb, sb


def scrolled_text(parent, **kwargs) -> tk.Text:
    sb = tk.Scrollbar(parent, bg=C_SURFACE)
    sb.pack(side=tk.RIGHT, fill=tk.Y)

    txt = tk.Text(
        parent,
        bg="#090910",
        fg=C_TEXT,
        font=("Courier", 9),
        relief=tk.FLAT,
        state=tk.DISABLED,
        padx=8,
        pady=6,
        yscrollcommand=sb.set,
        **kwargs,
    )
    txt.pack(fill=tk.BOTH, expand=True)
    sb.config(command=txt.yview)

    txt.tag_config("info",   foreground=C_TEXT_DIM)
    txt.tag_config("ok",     foreground=C_SUCCESS)
    txt.tag_config("err",    foreground="#f75a7c")
    txt.tag_config("folder", foreground=C_ACCENT)
    return txt


def log_append(txt_widget: tk.Text, msg: str, tag: str = "info"):
    txt_widget.config(state=tk.NORMAL)
    txt_widget.insert(tk.END, msg + "\n", tag)
    txt_widget.see(tk.END)
    txt_widget.config(state=tk.DISABLED)

def make_camera_canvas(parent) -> tk.Canvas:
    border = tk.Frame(parent, bg=C_BORDER, bd=1)
    border.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(border, bg=C_PANEL, highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    return canvas


def show_placeholder(canvas: tk.Canvas, text: str):
    canvas.delete("all")
    w = canvas.winfo_width() or 700
    h = canvas.winfo_height() or 480
    canvas.create_text(
        w // 2, h // 2,
        text=text,
        fill=C_TEXT_DIM,
        font=("Helvetica", 13),
        justify=tk.CENTER,
    )


def render_frame(canvas: tk.Canvas, frame):
    cw = canvas.winfo_width()
    ch = canvas.winfo_height()
    if cw < 4 or ch < 4:
        cw, ch = 700, 480

    h, w = frame.shape[:2]
    scale = min(cw / w, ch / h)
    nw, nh = int(w * scale), int(h * scale)
    rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)

    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
    canvas.delete("all")
    canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
    canvas._photo = photo   