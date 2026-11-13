from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..db import Database
from ..models import Node


def _node_choices(nodes: list[Node]) -> list[str]:
    return [f"{node.id} · {node.name}" for node in nodes]


def _choice_id(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value.split(" · ", 1)[0])
    except (ValueError, IndexError):
        return None


class FlowDialog(tk.Toplevel):
    def __init__(self, master, title: str = "新建流程", name: str = "", description: str = ""):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.result: dict | None = None
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="流程名称").grid(row=0, column=0, sticky="w", pady=4)
        self.name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(frame, textvariable=self.name_var, width=38)
        name_entry.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="说明").grid(row=1, column=0, sticky="nw", pady=4)
        self.desc = tk.Text(frame, width=38, height=5, wrap="word")
        self.desc.grid(row=1, column=1, sticky="ew", pady=4)
        self.desc.insert("1.0", description)
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="确定", command=self._ok).pack(side="right", padx=4)
        name_entry.focus_set()
        self.bind("<Return>", lambda _event: self._ok())
        self.transient(master)
        self.grab_set()

    def _ok(self):
        name = self.name_var.get().strip()
        if not name:
            return
        self.result = {"name": name, "description": self.desc.get("1.0", "end").strip()}
        self.destroy()


class ConnectDialog(tk.Toplevel):
    def __init__(
        self,
        master,
        nodes: list[Node],
        src: int | None = None,
        dst: int | None = None,
        weight: float = 1.0,
        label: str = "",
    ):
        super().__init__(master)
        self.title("连接步骤")
        self.resizable(False, False)
        self.result: dict | None = None
        choices = _node_choices(nodes)
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="当前步骤").grid(row=0, column=0, sticky="w", pady=4)
        self.src_var = tk.StringVar()
        src_combo = ttk.Combobox(frame, textvariable=self.src_var, values=choices, width=34, state="readonly")
        src_combo.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="下一步步骤").grid(row=1, column=0, sticky="w", pady=4)
        self.dst_var = tk.StringVar()
        ttk.Combobox(frame, textvariable=self.dst_var, values=choices, width=34, state="readonly").grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Label(frame, text="偏好权重").grid(row=2, column=0, sticky="w", pady=4)
        self.weight_var = tk.StringVar(value=str(weight))
        ttk.Spinbox(
            frame, from_=0.0, to=20.0, increment=0.1, textvariable=self.weight_var, width=10
        ).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="连线说明").grid(row=3, column=0, sticky="w", pady=4)
        self.label_var = tk.StringVar(value=label)
        ttk.Entry(frame, textvariable=self.label_var, width=34).grid(
            row=3, column=1, sticky="ew", pady=4
        )
        if src is not None:
            for choice in choices:
                if _choice_id(choice) == src:
                    self.src_var.set(choice)
        if dst is not None:
            for choice in choices:
                if _choice_id(choice) == dst:
                    self.dst_var.set(choice)
        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="连接", command=self._ok).pack(side="right", padx=4)
        self.transient(master)
        self.grab_set()

    def _ok(self):
        src = _choice_id(self.src_var.get())
        dst = _choice_id(self.dst_var.get())
        if src is None or dst is None:
            return
        try:
            weight = float(self.weight_var.get())
        except ValueError:
            weight = 1.0
        self.result = {
            "src": src,
            "dst": dst,
            "weight": weight,
            "label": self.label_var.get().strip(),
        }
        self.destroy()


class ImportDialog(tk.Toplevel):
    def __init__(
        self,
        master,
        db: Database,
        path: str,
        current_flow_id: int | None,
        current_nodes: list[Node],
    ):
        super().__init__(master)
        self.title("导入并衔接流程")
        self.resizable(False, False)
        self.path = path
        self.result: dict | None = None
        self.current_flow_id = current_flow_id
        choices = ["（不衔接）"] + _node_choices(current_nodes)
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text=f"文件：{path}", wraplength=420).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self.mode_var = tk.StringVar(value="merge" if current_flow_id else "new")
        mode_frame = ttk.LabelFrame(frame, text="导入方式", padding=8)
        mode_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Radiobutton(
            mode_frame, text="新建流程", variable=self.mode_var, value="new", command=self._toggle
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_frame,
            text="合并到当前流程",
            variable=self.mode_var,
            value="merge",
            command=self._toggle,
        ).pack(anchor="w")
        ttk.Label(frame, text="新流程名称").grid(row=2, column=0, sticky="w", pady=4)
        self.name_var = tk.StringVar()
        self.name_entry = ttk.Entry(frame, textvariable=self.name_var, width=34)
        self.name_entry.grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="上游衔接（当前节点 → 导入入口）").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.up_var = tk.StringVar(value="（不衔接）")
        self.up_combo = ttk.Combobox(
            frame, textvariable=self.up_var, values=choices, width=34, state="readonly"
        )
        self.up_combo.grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="下游衔接（导入出口 → 当前节点）").grid(
            row=4, column=0, sticky="w", pady=4
        )
        self.down_var = tk.StringVar(value="（不衔接）")
        self.down_combo = ttk.Combobox(
            frame, textvariable=self.down_var, values=choices, width=34, state="readonly"
        )
        self.down_combo.grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="衔接权重").grid(row=5, column=0, sticky="w", pady=4)
        self.weight_var = tk.StringVar(value="1.0")
        ttk.Spinbox(
            frame, from_=0.0, to=20.0, increment=0.1, textvariable=self.weight_var, width=10
        ).grid(row=5, column=1, sticky="w", pady=4)
        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="导入", command=self._ok).pack(side="right", padx=4)
        self._toggle()
        self.transient(master)
        self.grab_set()

    def _toggle(self):
        merge = self.mode_var.get() == "merge" and self.current_flow_id is not None
        state = "readonly" if merge else "disabled"
        self.up_combo.configure(state=state)
        self.down_combo.configure(state=state)
        self.name_entry.configure(state="disabled" if merge else "normal")

    def _ok(self):
        mode = self.mode_var.get()
        target = self.current_flow_id if mode == "merge" else None
        try:
            weight = float(self.weight_var.get())
        except ValueError:
            weight = 1.0
        self.result = {
            "target_flow_id": target,
            "new_flow_name": self.name_var.get().strip() or None,
            "link_from": _choice_id(self.up_var.get()) if target else None,
            "link_to": _choice_id(self.down_var.get()) if target else None,
            "link_weight": weight,
        }
        self.destroy()


class EdgeDialog(tk.Toplevel):
    def __init__(self, master, weight: float = 1.0, label: str = ""):
        super().__init__(master)
        self.title("编辑连线")
        self.resizable(False, False)
        self.result: dict | None = None
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="偏好权重").grid(row=0, column=0, sticky="w", pady=4)
        self.weight_var = tk.StringVar(value=str(weight))
        ttk.Spinbox(
            frame, from_=0.0, to=20.0, increment=0.1, textvariable=self.weight_var, width=12
        ).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="说明").grid(row=1, column=0, sticky="w", pady=4)
        self.label_var = tk.StringVar(value=label)
        ttk.Entry(frame, textvariable=self.label_var, width=30).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="保存", command=self._ok).pack(side="right", padx=4)
        self.transient(master)
        self.grab_set()

    def _ok(self):
        try:
            weight = float(self.weight_var.get())
        except ValueError:
            weight = 1.0
        self.result = {"weight": weight, "label": self.label_var.get().strip()}
        self.destroy()


class VariableDialog(tk.Toplevel):
    def __init__(self, master, name: str = "", value: str = ""):
        super().__init__(master)
        self.title("变量")
        self.resizable(False, False)
        self.result: dict | None = None
        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="变量名（可作为 Python 变量使用）").grid(
            row=0, column=0, sticky="w", pady=4
        )
        self.name_var = tk.StringVar(value=name)
        name_entry = ttk.Entry(frame, textvariable=self.name_var, width=32)
        name_entry.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="初始值").grid(row=1, column=0, sticky="w", pady=4)
        self.value_var = tk.StringVar(value=value)
        ttk.Entry(frame, textvariable=self.value_var, width=32).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        ttk.Label(
            frame,
            text="数字、true/false、[1,2] 会按 JSON 解析；其他按字符串。",
            foreground="#64748b",
            wraplength=330,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(2, 6))
        buttons = ttk.Frame(frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="right", padx=4)
        ttk.Button(buttons, text="确定", command=self._ok).pack(side="right", padx=4)
        name_entry.focus_set()
        self.bind("<Return>", lambda _event: self._ok())
        self.transient(master)
        self.grab_set()

    def _ok(self):
        name = self.name_var.get().strip()
        if not name:
            return
        self.result = {"name": name, "value": self.value_var.get()}
        self.destroy()
