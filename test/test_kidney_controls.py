"""Exercise the panel's actual callbacks without starting a desktop or Tk window."""

from types import SimpleNamespace

import pytest

import probe_tracking.kidney_controls as controls_module
from probe_tracking.kidney_controls import KidneyControls


class TkError(Exception):
    pass


class Variable:
    def __init__(self, *, master, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self, parent=None, **options):
        self.options = options
        self.bindings = {}

    def grid(self, **options):
        pass

    def columnconfigure(self, index, **options):
        pass

    def rowconfigure(self, index, **options):
        pass

    def bind(self, event, callback):
        self.bindings[event] = callback


class Root(Widget):
    def __init__(self):
        super().__init__()
        self.events = []
        self.protocols = {}

    def title(self, value):
        pass

    def resizable(self, width, height):
        pass

    def minsize(self, width, height):
        pass

    def geometry(self, value):
        pass

    def protocol(self, name, callback):
        self.protocols[name] = callback

    def update_idletasks(self):
        self.events.append("idle")

    def update(self):
        self.events.append("update")

    def destroy(self):
        self.events.append("destroy")


@pytest.fixture
def panel(monkeypatch):
    root = Root()
    widgets = []

    def widget(parent, **options):
        result = Widget(parent, **options)
        widgets.append(result)
        return result

    monkeypatch.setattr(
        controls_module, "tk", SimpleNamespace(Tk=lambda: root, StringVar=Variable, DoubleVar=Variable, TclError=TkError)
    )
    monkeypatch.setattr(
        controls_module, "ttk",
        SimpleNamespace(**dict.fromkeys(("Frame", "Label", "LabelFrame", "Entry", "Button", "Scale", "Combobox"), widget)),
    )
    changed, reset, stopped = [], [], []
    controls = KidneyControls(
        position_mm=(12, -31, 402), scale=1.3,
        on_change=lambda position, scale: changed.append((position, scale)),
        on_reset_history=lambda: reset.append(True), on_stop=lambda: stopped.append(True),
    )
    return SimpleNamespace(controls=controls, root=root, widgets=widgets, changed=changed, reset=reset, stopped=stopped)


def click(panel, label, index=0):
    matching = [widget for widget in panel.widgets if widget.options.get("text") == label]
    matching[index].options["command"]()


def test_position_buttons_preserve_other_axes_and_use_selected_step(panel):
    assert not panel.changed
    click(panel, "＋", 0)
    assert panel.changed[-1] == ((17, -31, 402), 1.3)
    panel.controls.step_var.set("25")
    click(panel, "−", 1)
    click(panel, "＋", 2)
    assert panel.changed[-1] == ((17, -56, 427), 1.3)
    assert [var.get() for var in panel.controls.position_vars] == ["17", "-56", "427"]


def test_position_entries_apply_on_enter_and_button(panel):
    for variable, value in zip(panel.controls.position_vars, ("1.5", "-52", "450"), strict=True):
        variable.set(value)
    entry = next(widget for widget in panel.widgets if "<Return>" in widget.bindings)
    entry.bindings["<Return>"](None)
    assert panel.changed == [((1.5, -52, 450), 1.3)]
    panel.controls.position_vars[2].set("600")
    click(panel, "位置を適用")
    assert panel.changed[-1] == ((1.5, -52, 600), 1.3)


@pytest.mark.parametrize("invalid", ["", "oops", "nan", "inf", "-inf", "1e999"])
def test_invalid_position_never_calls_change(panel, invalid):
    panel.controls.position_vars[1].set(invalid)
    click(panel, "位置を適用")
    assert not panel.changed
    assert "有限" in panel.controls.status_var.get()


def test_slider_numeric_entry_and_scale_buttons_change_real_scale(panel):
    slider = next(widget for widget in panel.widgets if "from_" in widget.options)
    slider.options["command"]("177.8")
    assert panel.changed[-1] == ((12, -31, 402), 1.78)
    assert panel.controls.scale_var.get() == "178"
    panel.controls.scale_var.set("125")
    click(panel, "適用")
    assert panel.changed[-1] == ((12, -31, 402), 1.25)
    click(panel, "＋10 %")
    assert panel.changed[-1][1] == 1.35
    click(panel, "−10 %")
    assert panel.changed[-1][1] == 1.25


@pytest.mark.parametrize("invalid", ["", "oops", "nan", "inf", "-inf", "0", "9.9", "500.1", "1e999"])
def test_invalid_scale_has_inline_message_and_does_not_change_placement(panel, invalid):
    panel.controls.scale_var.set(invalid)
    click(panel, "適用")
    assert not panel.changed
    assert "10〜500" in panel.controls.status_var.get()


@pytest.mark.parametrize("percent,label,scale", [("10", "−10 %", 0.1), ("500", "＋10 %", 5)])
def test_scale_buttons_stop_at_bounds_without_duplicate_callback(panel, percent, label, scale):
    panel.controls.scale_var.set(percent)
    click(panel, "適用")
    click(panel, label)
    assert panel.changed == [((12, -31, 402), scale)]


def test_reset_restores_custom_startup_values_without_clearing_history(panel):
    click(panel, "＋", 0)
    click(panel, "＋10 %")
    panel.controls.position_vars[0].set("invalid")
    click(panel, "起動時の位置・大きさに戻す")
    assert panel.changed[-1] == ((12, -31, 402), 1.3)
    assert panel.controls.position_vars[0].get() == "12"
    assert not panel.reset
    before = panel.changed.copy()
    click(panel, "着色履歴を消去")
    assert panel.reset == [True]
    assert panel.changed == before


def test_invalid_step_does_not_move(panel):
    panel.controls.step_var.set("nan")
    click(panel, "＋", 0)
    assert not panel.changed
    assert "移動量" in panel.controls.status_var.get()


@pytest.mark.parametrize("stop_method", ["window", "button"])
def test_window_close_and_stop_button_stop_once_and_pump_is_safe_after_close(panel, stop_method):
    panel.controls.pump()
    assert panel.root.events == ["idle", "update"]
    if stop_method == "window":
        panel.root.protocols["WM_DELETE_WINDOW"]()
    else:
        click(panel, "停止")
    panel.controls.pump()
    panel.controls.close()
    panel.root.protocols["WM_DELETE_WINDOW"]()
    assert panel.controls.closed
    assert panel.stopped == [True]
    assert panel.root.events == ["idle", "update", "destroy"]


def test_external_close_is_idempotent_and_does_not_request_stop(panel):
    panel.controls.close()
    panel.controls.close()
    assert panel.controls.closed
    assert panel.root.events == ["destroy"]
    assert not panel.stopped


def test_pump_handles_window_destroyed_by_tk(panel):
    def destroyed():
        raise TkError("application has been destroyed")

    panel.root.update = destroyed
    panel.root.destroy = destroyed
    panel.controls.pump()
    assert panel.controls.closed
    assert panel.stopped == [True]


@pytest.mark.parametrize("missing_module", [False, True])
def test_tk_unavailable_has_actionable_error(monkeypatch, missing_module):
    def unavailable():
        raise TkError("no display")

    monkeypatch.setattr(controls_module, "tk", None if missing_module else SimpleNamespace(Tk=unavailable, TclError=TkError))
    with pytest.raises(RuntimeError, match="--no-kidney-controls"):
        KidneyControls(position_mm=(0, 0, 0), scale=1, on_change=lambda *_: None,
                       on_reset_history=lambda: None, on_stop=lambda: None)


@pytest.mark.parametrize("fails", [False, True])
def test_tk_initialization_preserves_python_interrupt_handler(panel, monkeypatch, fails):
    original_handler = object()
    native_handler = [original_handler]
    sigint = controls_module.signal.SIGINT

    def create_root():
        # Tk replaces the native handler while Python's getsignal() still
        # reports its old cached handler; simulate both views independently.
        native_handler[0] = object()
        if fails:
            raise TkError("no display")
        return panel.root

    def restore_signal(signum, handler):
        assert signum == sigint
        native_handler[0] = handler

    monkeypatch.setattr(controls_module.tk, "Tk", create_root)
    monkeypatch.setattr(
        controls_module, "signal",
        SimpleNamespace(SIGINT=sigint, getsignal=lambda signum: original_handler, signal=restore_signal),
    )
    options = dict(position_mm=(0, 0, 0), scale=1, on_change=lambda *_: None,
                   on_reset_history=lambda: None, on_stop=lambda: None)
    if fails:
        with pytest.raises(RuntimeError, match="--no-kidney-controls"):
            KidneyControls(**options)
    else:
        controls = KidneyControls(**options)
        controls.close()
    assert native_handler[0] is original_handler
