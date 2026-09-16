"""A small native panel for changing kidney placement while tracking runs."""

import signal
from collections.abc import Callable, Sequence
from math import isfinite

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:  # Headless installations can still track without this panel.
    tk = None
    ttk = None


class KidneyControls:
    """Pump on the main thread; callbacks only request the next frame's edits."""

    def __init__(
        self,
        *,
        position_mm: Sequence[float],
        scale: float,
        on_change: Callable[[tuple[float, float, float], float], None],
        on_reset_history: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> None:
        position = tuple(float(value) for value in position_mm)
        if len(position) != 3 or not all(isfinite(value) for value in position):
            raise ValueError("Kidney position must contain three finite coordinates")
        scale = float(scale)
        if not isfinite(scale) or not 0.1 <= scale <= 5.0:
            raise ValueError("Kidney scale must be between 0.1 and 5.0")
        self._initial_position = self._position = position
        self._initial_scale = self._scale = scale
        self._on_change = on_change
        self._on_reset_history = on_reset_history
        self._on_stop = on_stop
        self._closed = False
        self._syncing = False
        unavailable = (
            "Kidney controls require a desktop session and Python with Tk support. "
            "Use --no-kidney-controls to run without the control panel."
        )
        if tk is None:
            raise RuntimeError(unavailable)
        interrupt_handler = signal.getsignal(signal.SIGINT)
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            raise RuntimeError(unavailable) from error
        finally:
            # macOS Tk 9 replaces the C SIGINT handler without updating Python's
            # getsignal() state. Reinstall it so Ctrl+C reaches loop cleanup.
            signal.signal(signal.SIGINT, interrupt_handler)
        self.root.title("腎臓の位置・大きさ")
        self.root.resizable(True, True)
        self.root.minsize(640, 550)
        self.root.geometry("650x590")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._request_stop)

        self.position_vars = [tk.StringVar(master=self.root, value=f"{value:g}") for value in position]
        self.step_var = tk.StringVar(master=self.root, value="5")
        self.scale_var = tk.StringVar(master=self.root, value=f"{scale * 100:g}")
        self.slider_var = tk.DoubleVar(master=self.root, value=scale * 100)
        self.status_var = tk.StringVar(master=self.root, value="位置・大きさは操作するとすぐに反映されます。")
        self._build_panel()

    @property
    def closed(self) -> bool:
        return self._closed

    def _build_panel(self) -> None:
        panel = ttk.Frame(self.root, padding=16)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="腎臓の位置・大きさ", font=("", 18, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(panel, text="カメラ座標：X は右、Y は下、Z は前方が＋（単位 mm）").grid(
            row=1, column=0, sticky="w", pady=(6, 10)
        )

        position = ttk.LabelFrame(panel, text="腎臓中心の位置", padding=10)
        position.grid(row=2, column=0, sticky="ew")
        position.columnconfigure(1, weight=1)
        ttk.Label(position, text="移動量 (mm)").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Combobox(
            position, textvariable=self.step_var, values=("1", "5", "10", "25"), state="readonly", width=8
        ).grid(row=0, column=1, sticky="w", pady=(0, 5))
        for axis, label in enumerate(("X（右）", "Y（下）", "Z（前方）")):
            ttk.Label(position, text=label).grid(row=axis + 1, column=0, sticky="w", padx=(0, 10))
            entry = ttk.Entry(position, textvariable=self.position_vars[axis], width=14)
            entry.grid(row=axis + 1, column=1, sticky="ew", pady=3)
            entry.bind("<Return>", lambda _event: self._apply_position())
            for column, direction, text in ((2, -1, "−"), (3, 1, "＋")):
                ttk.Button(
                    position, text=text, width=4, command=lambda a=axis, d=direction: self._move(a, d)
                ).grid(row=axis + 1, column=column, padx=(6, 0))
        ttk.Button(position, text="位置を適用", command=self._apply_position).grid(
            row=4, column=0, columnspan=4, sticky="e", pady=(5, 0)
        )

        scale = ttk.LabelFrame(panel, text="腎臓モデルの大きさ", padding=10)
        scale.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        scale.columnconfigure(1, weight=1)
        ttk.Label(scale, text="10 %").grid(row=0, column=0)
        ttk.Scale(
            scale, from_=10, to=500, variable=self.slider_var, command=self._slide_scale
        ).grid(row=0, column=1, columnspan=3, sticky="ew", padx=8)
        ttk.Label(scale, text="500 %").grid(row=0, column=4)
        entry = ttk.Entry(scale, textvariable=self.scale_var, width=9)
        entry.grid(row=1, column=0, pady=(7, 0))
        entry.bind("<Return>", lambda _event: self._apply_scale())
        ttk.Label(scale, text="%（100 % = 元の大きさ）").grid(row=1, column=1, sticky="w", padx=6, pady=(7, 0))
        ttk.Button(scale, text="適用", width=6, command=self._apply_scale).grid(row=1, column=2, pady=(7, 0))
        ttk.Button(scale, text="−10 %", width=7, command=lambda: self._change_scale(-0.1)).grid(
            row=1, column=3, padx=(6, 0), pady=(7, 0)
        )
        ttk.Button(scale, text="＋10 %", width=7, command=lambda: self._change_scale(0.1)).grid(
            row=1, column=4, padx=(6, 0), pady=(7, 0)
        )
        ttk.Label(
            panel,
            text="大きさは腎臓の実寸と交差判定に反映されます。\n着色履歴は腎臓と一緒に移動・拡大縮小します。",
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(panel)
        buttons.grid(row=5, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="起動時の位置・大きさに戻す", command=self._reset_placement).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(buttons, text="着色履歴を消去", command=self._reset_history).grid(row=0, column=1)
        ttk.Button(buttons, text="停止", command=self._request_stop).grid(row=0, column=2, padx=(8, 0))
        ttk.Label(panel, textvariable=self.status_var, wraplength=540, justify="left").grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )

    def _publish(self, position: tuple[float, float, float], scale: float) -> None:
        if self._closed:
            return
        changed = position != self._position or scale != self._scale
        self._position, self._scale = position, scale
        self._syncing = True
        try:
            for variable, value in zip(self.position_vars, position, strict=True):
                variable.set(f"{value:g}")
            self.scale_var.set(f"{scale * 100:g}")
            self.slider_var.set(scale * 100)
        finally:
            self._syncing = False
        self.status_var.set(
            f"位置 ({position[0]:g}, {position[1]:g}, {position[2]:g}) mm ／ 大きさ {scale * 100:g} %"
        )
        if changed:
            self._on_change(position, scale)

    def _apply_position(self) -> None:
        try:
            position = tuple(float(variable.get()) for variable in self.position_vars)
            if not all(isfinite(value) for value in position):
                raise ValueError
        except (ValueError, OverflowError):
            self.status_var.set("位置には有限の数値を入力してください。")
            return
        self._publish(position, self._scale)

    def _move(self, axis: int, direction: int) -> None:
        try:
            step = float(self.step_var.get())
            if step not in (1, 5, 10, 25):
                raise ValueError
        except ValueError:
            self.status_var.set("移動量は 1、5、10、25 mm から選んでください。")
            return
        position = list(self._position)
        position[axis] += direction * step
        self._publish(tuple(position), self._scale)

    def _apply_scale(self) -> None:
        try:
            scale = float(self.scale_var.get()) / 100
            if not isfinite(scale) or not 0.1 <= scale <= 5:
                raise ValueError
        except (ValueError, OverflowError):
            self.status_var.set("大きさには 10〜500 % の数値を入力してください。")
            return
        self._publish(self._position, scale)

    def _slide_scale(self, percent: str) -> None:
        if self._syncing or self._closed:
            return
        # Whole percentages keep the numeric entry stable while dragging.
        scale = min(5.0, max(0.1, round(float(percent)) / 100))
        self._publish(self._position, scale)

    def _change_scale(self, amount: float) -> None:
        self._publish(self._position, min(5.0, max(0.1, round(self._scale + amount, 10))))

    def _reset_placement(self) -> None:
        self._publish(self._initial_position, self._initial_scale)

    def _reset_history(self) -> None:
        if not self._closed:
            self._on_reset_history()
            self.status_var.set("着色履歴を消去しました。現在の交差表示は続きます。")

    def _request_stop(self) -> None:
        if not self._closed:
            self.close()
            self._on_stop()

    def pump(self) -> None:
        """Process pending window events without taking over the tracking loop."""
        if not self._closed:
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self._request_stop()

    def close(self) -> None:
        """Release the window, including when the loop exits for another reason."""
        if not self._closed:
            self._closed = True
            try:
                self.root.destroy()
            except tk.TclError:
                pass
