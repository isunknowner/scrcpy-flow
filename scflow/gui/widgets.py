from __future__ import annotations

import base64
import io
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from PIL import Image

try:
    from PIL import ImageTk
except Exception:
    ImageTk = None

from ..embeddings import to_pil


def photo_image(image: Image.Image):
    if ImageTk is not None:
        return ImageTk.PhotoImage(image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return tk.PhotoImage(data=base64.b64encode(buffer.getvalue()))


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _bind_wheel(self, _event=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _event=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _wheel(self, event):
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta * 2, "units")


class ImageCanvas(tk.Canvas):
    def __init__(
        self,
        master,
        width: int = 300,
        height: int = 520,
        mode: str = "view",
        on_click: Callable[[float, float], None] | None = None,
        on_roi: Callable[[tuple[float, float, float, float]], None] | None = None,
        on_target: Callable[[tuple[float, float, float, float]], None] | None = None,
        on_zoom: Callable[[float], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            background="#0e1116",
            highlightthickness=0,
            **kwargs,
        )
        self.mode = mode
        self.on_click = on_click
        self.on_roi = on_roi
        self.on_target = on_target
        self.on_zoom = on_zoom
        self._image: Image.Image | None = None
        self._photo = None
        self._click: tuple[float, float] | None = None
        self._roi: tuple[float, float, float, float] | None = None
        self._target: tuple[float, float, float, float] | None = None
        self._geometry: tuple[int, int, int, int] | None = None
        self._zoom = 1.0
        self._zoom_min = 1.0
        self._zoom_max = 8.0
        self._drag_kind: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_now: tuple[float, float] | None = None
        self._canvas_size = (width, height)
        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Control-MouseWheel>", self._on_zoom_wheel)
        self.bind("<Control-Button-4>", self._on_zoom_wheel)
        self.bind("<Control-Button-5>", self._on_zoom_wheel)
        self.bind("<ButtonPress-2>", lambda event: self.scan_mark(event.x, event.y))
        self.bind("<B2-Motion>", lambda event: self.scan_dragto(event.x, event.y, gain=1))
        self._update_cursor()

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self._update_cursor()
        self.redraw()

    def _update_cursor(self) -> None:
        cursors = {"click": "crosshair", "roi": "tcross", "target": "crosshair", "view": "arrow"}
        self.configure(cursor=cursors.get(self.mode, "arrow"))

    def set_source(self, source: Any) -> None:
        if source is None:
            self._image = None
        else:
            try:
                self._image = to_pil(source).copy()
            except Exception:
                self._image = None
        self.redraw()

    def set_markers(
        self,
        click: tuple[float, float] | None = None,
        roi: tuple[float, float, float, float] | None = None,
        target: tuple[float, float, float, float] | None = None,
    ) -> None:
        self._click = click
        self._roi = roi
        self._target = target
        self.redraw()

    def set_zoom(self, zoom: float, focus: tuple[float, float] | None = None) -> None:
        new_zoom = min(self._zoom_max, max(self._zoom_min, float(zoom)))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        if focus is None and self._geometry:
            geom = self._geometry
            width = max(1, self.winfo_width())
            height = max(1, self.winfo_height())
            cx = self.canvasx(width / 2)
            cy = self.canvasy(height / 2)
            focus = (
                min(1.0, max(0.0, (cx - geom[0]) / max(1, geom[2]))),
                min(1.0, max(0.0, (cy - geom[1]) / max(1, geom[3]))),
            )
        if focus is None:
            focus = (0.5, 0.5)
        self._zoom = new_zoom
        self.redraw()
        self.after_idle(lambda: self._center_on(focus[0], focus[1]))
        if self.on_zoom:
            self.on_zoom(self._zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.3)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.3)

    def zoom_fit(self) -> None:
        self._zoom = 1.0
        self.redraw()
        self.after_idle(lambda: (self.xview_moveto(0), self.yview_moveto(0)))
        if self.on_zoom:
            self.on_zoom(self._zoom)

    def _center_on(self, rel_x: float, rel_y: float) -> None:
        if not self._geometry:
            return
        left, top, width, height = self._geometry
        scroll_w, scroll_h = self._scroll_size()
        view_w = max(1, self.winfo_width())
        view_h = max(1, self.winfo_height())
        target_x = left + rel_x * width - view_w / 2
        target_y = top + rel_y * height - view_h / 2
        self.xview_moveto(max(0.0, min(1.0, target_x / max(1, scroll_w))))
        self.yview_moveto(max(0.0, min(1.0, target_y / max(1, scroll_h))))

    def _scroll_size(self) -> tuple[int, int]:
        try:
            values = str(self.cget("scrollregion")).split()
            return int(float(values[2])), int(float(values[3]))
        except (ValueError, IndexError, tk.TclError):
            return max(1, self.winfo_width()), max(1, self.winfo_height())

    def _on_resize(self, event):
        self._canvas_size = (max(1, event.width), max(1, event.height))
        self.redraw()

    def _to_relative(self, x: float, y: float) -> tuple[float, float] | None:
        if not self._geometry:
            return None
        left, top, width, height = self._geometry
        if not (left <= x <= left + width and top <= y <= top + height):
            return None
        rel_x = min(1.0, max(0.0, (x - left) / width))
        rel_y = min(1.0, max(0.0, (y - top) / height))
        return rel_x, rel_y

    def _on_press(self, event):
        relative = self._to_relative(self.canvasx(event.x), self.canvasy(event.y))
        if relative is None:
            return
        if self.mode == "click":
            self._click = relative
            self.redraw()
            if self.on_click:
                self.on_click(relative[0], relative[1])
        elif self.mode in ("roi", "target"):
            self._drag_kind = self.mode
            self._drag_start = relative
            self._drag_now = relative
            self.redraw()

    def _on_drag(self, event):
        if self._drag_kind is None or self._drag_start is None:
            return
        relative = self._to_relative(self.canvasx(event.x), self.canvasy(event.y))
        if relative:
            self._drag_now = relative
            self.redraw()

    def _on_release(self, event):
        if self._drag_kind is None or self._drag_start is None:
            return
        relative = self._to_relative(self.canvasx(event.x), self.canvasy(event.y)) or self._drag_now
        start = self._drag_start
        kind = self._drag_kind
        self._drag_start = None
        self._drag_now = None
        self._drag_kind = None
        if not relative:
            self.redraw()
            return
        x1, y1 = start
        x2, y2 = relative
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left < 0.02 or bottom - top < 0.02:
            self.redraw()
            return
        region = (left, top, right - left, bottom - top)
        if kind == "target":
            self._target = region
            self.redraw()
            if self.on_target:
                self.on_target(region)
        else:
            self._roi = region
            self.redraw()
            if self.on_roi:
                self.on_roi(region)

    def _on_zoom_wheel(self, event):
        if self._image is None:
            return "break"
        delta = getattr(event, "delta", 0)
        num = getattr(event, "num", None)
        if num == 4 or delta > 0:
            factor = 1.2
        else:
            factor = 1 / 1.2
        focus = self._to_relative(self.canvasx(event.x), self.canvasy(event.y))
        self.set_zoom(self._zoom * factor, focus=focus)
        return "break"

    def redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        if self._image is None:
            self.create_text(
                width // 2,
                height // 2,
                text="未设置图片\n可在编辑模式导入或从手机截图",
                fill="#6b7684",
                justify="center",
                font=("TkDefaultFont", 11),
            )
            self._geometry = None
            self.configure(scrollregion=(0, 0, width, height))
            return
        image_w, image_h = self._image.size
        padding = 8
        base_scale = min(
            (width - padding * 2) / image_w,
            (height - padding * 2) / image_h,
        )
        scale = max(0.01, base_scale * self._zoom)
        draw_w = max(1, round(image_w * scale))
        draw_h = max(1, round(image_h * scale))
        if draw_w <= width - padding * 2 and draw_h <= height - padding * 2:
            left = (width - draw_w) // 2
            top = (height - draw_h) // 2
        else:
            left = padding
            top = padding
        scroll_w = max(width, draw_w + padding * 2)
        scroll_h = max(height, draw_h + padding * 2)
        self.configure(scrollregion=(0, 0, scroll_w, scroll_h))
        self._geometry = (left, top, draw_w, draw_h)
        resized = self._image.resize((draw_w, draw_h), Image.Resampling.LANCZOS)
        self._photo = photo_image(resized)
        self.create_image(left, top, image=self._photo, anchor="nw")
        if self._roi:
            rx, ry, rw, rh = self._roi
            x1 = left + rx * draw_w
            y1 = top + ry * draw_h
            x2 = x1 + rw * draw_w
            y2 = y1 + rh * draw_h
            self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline="#4cb1ff",
                width=2,
                dash=(6, 3),
                fill="#4cb1ff",
                stipple="gray25",
            )
            self.create_text(
                x1 + 4,
                max(top + 8, y1 - 10),
                text="检测重点",
                fill="#4cb1ff",
                anchor="w",
                font=("TkDefaultFont", 9, "bold"),
            )
        if self._target:
            tx, ty, tw, th = self._target
            x1 = left + tx * draw_w
            y1 = top + ty * draw_h
            x2 = x1 + tw * draw_w
            y2 = y1 + th * draw_h
            self.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline="#ff9f1c",
                width=2,
                dash=(4, 2),
                fill="#ff9f1c",
                stipple="gray12",
            )
            self.create_text(
                x1 + 4,
                min(top + draw_h - 12, y2 + 10),
                text="点击对象",
                fill="#ff9f1c",
                anchor="w",
                font=("TkDefaultFont", 9, "bold"),
            )
        if self._drag_start and self._drag_now:
            x1 = left + min(self._drag_start[0], self._drag_now[0]) * draw_w
            y1 = top + min(self._drag_start[1], self._drag_now[1]) * draw_h
            x2 = left + max(self._drag_start[0], self._drag_now[0]) * draw_w
            y2 = top + max(self._drag_start[1], self._drag_now[1]) * draw_h
            color = "#ff9f1c" if self._drag_kind == "target" else "#ffd166"
            self.create_rectangle(x1, y1, x2, y2, outline=color, width=2, dash=(4, 2))
        if self._click:
            cx = left + self._click[0] * draw_w
            cy = top + self._click[1] * draw_h
            self.create_line(cx - 18, cy, cx + 18, cy, fill="#ff5c5c", width=2)
            self.create_line(cx, cy - 18, cx, cy + 18, fill="#ff5c5c", width=2)
            self.create_oval(cx - 11, cy - 11, cx + 11, cy + 11, outline="#ff5c5c", width=2)


class ScreenshotViewer(tk.Toplevel):
    def __init__(self, master, image: Any, title: str = "截图预览"):
        super().__init__(master)
        self.title(title)
        self.geometry("420x820")
        canvas = ImageCanvas(self, width=400, height=780, mode="view")
        canvas.pack(fill="both", expand=True, padx=6, pady=6)
        canvas.set_source(image)
        self.bind("<Escape>", lambda _event: self.destroy())


class FallbackStepDialog(tk.Toplevel):
    TYPES = [
        ("key", "按键"),
        ("wait", "等待"),
        ("tap", "点击"),
        ("swipe", "滑动"),
        ("text", "输入文本"),
    ]

    def __init__(self, master, step: dict | None = None):
        super().__init__(master)
        self.title("保底步骤")
        self.resizable(False, False)
        self.result: dict | None = None
        self.step = dict(step or {"type": "key", "key": "back", "wait": 0.8})
        self.type_var = tk.StringVar(value=self.step.get("type", "key"))
        self.key_var = tk.StringVar(value=str(self.step.get("key", "back")))
        self.text_var = tk.StringVar(value=str(self.step.get("text", "")))
        self.wait_var = tk.StringVar(value=str(self.step.get("wait", self.step.get("seconds", 0.8))))
        self.coords = {}
        for name, default in (
            ("x", 0.5),
            ("y", 0.5),
            ("x1", 0.5),
            ("y1", 0.8),
            ("x2", 0.5),
            ("y2", 0.2),
            ("duration", 300),
        ):
            self.coords[name] = tk.StringVar(value=str(self.step.get(name, default)))

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="类型").grid(row=0, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(
            frame,
            textvariable=self.type_var,
            values=[value for value, _ in self.TYPES],
            state="readonly",
            width=16,
        )
        combo.grid(row=0, column=1, sticky="w", pady=4)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._rebuild())

        self.dynamic = ttk.Frame(frame)
        self.dynamic.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frame, text="之后等待(秒)").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.wait_var, width=12).grid(
            row=2, column=1, sticky="w", pady=4
        )

        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="确定", command=self._ok).pack(side="right", padx=4)
        self._rebuild()
        self.transient(master)
        self.grab_set()

    def _rebuild(self):
        for child in self.dynamic.winfo_children():
            child.destroy()
        kind = self.type_var.get()
        row = 0
        if kind == "key":
            ttk.Label(self.dynamic, text="按键").grid(row=row, column=0, sticky="w")
            ttk.Combobox(
                self.dynamic,
                textvariable=self.key_var,
                values=["back", "home", "enter", "menu", "power", "volume_up", "volume_down"],
                width=14,
            ).grid(row=row, column=1, sticky="w")
        elif kind == "text":
            ttk.Label(self.dynamic, text="文本").grid(row=row, column=0, sticky="w")
            ttk.Entry(self.dynamic, textvariable=self.text_var, width=24).grid(
                row=row, column=1, sticky="w"
            )
        elif kind == "tap":
            for name, label in (("x", "x 比例"), ("y", "y 比例")):
                ttk.Label(self.dynamic, text=label).grid(row=row, column=0, sticky="w")
                ttk.Entry(self.dynamic, textvariable=self.coords[name], width=10).grid(
                    row=row, column=1, sticky="w"
                )
                row += 1
        elif kind == "swipe":
            for name, label in (
                ("x1", "起点 x"),
                ("y1", "起点 y"),
                ("x2", "终点 x"),
                ("y2", "终点 y"),
                ("duration", "时长 ms"),
            ):
                ttk.Label(self.dynamic, text=label).grid(row=row, column=0, sticky="w")
                ttk.Entry(self.dynamic, textvariable=self.coords[name], width=10).grid(
                    row=row, column=1, sticky="w"
                )
                row += 1
        elif kind == "wait":
            ttk.Label(self.dynamic, text="等待秒数").grid(row=row, column=0, sticky="w")
            ttk.Entry(self.dynamic, textvariable=self.wait_var, width=10).grid(
                row=row, column=1, sticky="w"
            )

    def _ok(self):
        kind = self.type_var.get()
        step = {"type": kind}
        try:
            step["wait"] = float(self.wait_var.get())
            if kind == "key":
                step["key"] = self.key_var.get().strip() or "back"
            elif kind == "text":
                step["text"] = self.text_var.get()
            elif kind == "tap":
                step["x"] = float(self.coords["x"].get())
                step["y"] = float(self.coords["y"].get())
            elif kind == "swipe":
                for name in ("x1", "y1", "x2", "y2"):
                    step[name] = float(self.coords[name].get())
                step["duration"] = int(float(self.coords["duration"].get()))
            elif kind == "wait":
                step["seconds"] = step.pop("wait")
        except ValueError:
            messagebox.showerror("参数错误", "请检查数字参数", parent=self)
            return
        self.result = step
        self.destroy()


class TextDialog(tk.Toplevel):
    def __init__(self, master, title: str, label: str, initial: str = ""):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result: str | None = None
        frame = ttk.Frame(self, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=label).pack(anchor="w")
        self.var = tk.StringVar(value=initial)
        entry = ttk.Entry(frame, textvariable=self.var, width=36)
        entry.pack(fill="x", pady=8)
        entry.focus_set()
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="确定", command=self._ok).pack(side="right", padx=4)
        self.bind("<Return>", lambda _event: self._ok())
        self.transient(master)
        self.grab_set()

    def _ok(self):
        self.result = self.var.get().strip()
        self.destroy()
