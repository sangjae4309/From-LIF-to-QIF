#!/usr/bin/env python
"""Run pinnpy.py with monkeypatched debug wrappers for selected pinn.py functions."""

from __future__ import annotations

import functools
import inspect
import itertools
import os
import runpy
import sys
import threading
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

import jax


PINN_POISSON_DIR = Path(__file__).resolve().parent
PINNPY_FILE = PINN_POISSON_DIR / "pinnpy.py"
DUMP_FILE = PINN_POISSON_DIR / "dump.log"
DUMP_CALL_COUNTER = itertools.count(1)
DUMP_LOCK = threading.Lock()
CALL_STACK = threading.local()

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Edit this mapping per debugging workload. Names matching function arguments are
# dumped on call. Other names are mapped to return tuple items in order.
TARGET_VARIABLES_BY_FUNCTION = {
    "simulatefn": ["input"],
    "lossfn": ["input_physics", "d2u_dt2", "d2u_dx2", "pde_residual", "loss"],
    "compute_derivatives": ["input", "du_dt", "d2u_dt2"],
}


def summarize(value: Any, *, depth: int = 0) -> str:
    """Return a compact summary that is safe for JAX arrays and nested params."""
    if depth > 2:
        return "..."

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return f"{type(value).__name__}(shape={tuple(shape)}, dtype={dtype})"

    if isinstance(value, dict):
        items = list(value.items())
        preview = {k: summarize(v, depth=depth + 1) for k, v in items[:8]}
        suffix = "" if len(items) <= 8 else f", ... +{len(items) - 8} keys"
        return f"dict({pformat(preview)}{suffix})"

    if isinstance(value, (list, tuple)):
        items = [summarize(v, depth=depth + 1) for v in value[:6]]
        suffix = "" if len(value) <= 6 else f", ... +{len(value) - 6} items"
        return f"{type(value).__name__}([{', '.join(items)}{suffix}])"

    text = repr(value)
    if len(text) > 240:
        text = text[:237] + "..."
    return f"{type(value).__name__}({text})"


def summarize_concrete(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    try:
        text = repr(value.tolist())
    except AttributeError:
        text = repr(value)
    if len(text) > 1000:
        text = text[:997] + "..."
    if shape is not None:
        return f"{type(value).__name__}(shape={tuple(shape)}, dtype={dtype}, value={text})"
    return f"{type(value).__name__}({text})"


def append_dump(text: str) -> None:
    with DUMP_LOCK, DUMP_FILE.open("a") as dump:
        dump.write(text)
        dump.write("\n")


def is_callback_value(value: Any) -> bool:
    if hasattr(value, "shape") or isinstance(value, (int, float, bool, complex)):
        return True
    return False


def current_stack() -> list[int]:
    stack = getattr(CALL_STACK, "value", None)
    if stack is None:
        stack = []
        CALL_STACK.value = stack
    return stack


def write_event(
    call_id: int,
    depth: int,
    phase: str,
    function_name: str,
    source_file: str | None,
    entries: list[tuple[str, str]],
    *,
    kind: str,
) -> None:
    indent = "  " * depth
    arrow = "-->" if phase == "call" else "<--"
    lines = [f"{indent}{arrow} #{call_id:06d} {function_name} [{kind}]"]
    if phase == "call":
        lines.append(f"{indent}    source: {source_file}")
    for name, summary in entries:
        lines.append(f"{indent}    {name} = {summary}")
    append_dump("\n".join(lines))


def dump_event(
    call_id: int,
    depth: int,
    phase: str,
    function_name: str,
    source_file: str | None,
    entries: list[tuple[str, Any]],
) -> None:
    if not entries:
        return

    callback_values = []
    metadata = []
    for name, value in entries:
        if is_callback_value(value):
            metadata.append((name, None))
            callback_values.append(value)
        else:
            metadata.append((name, summarize(value)))

    def callback(*concrete_values: Any) -> None:
        concrete_iter = iter(concrete_values)
        summarized_entries = []
        for name, static_summary in metadata:
            if static_summary is None:
                summarized_entries.append((name, summarize_concrete(next(concrete_iter))))
            else:
                summarized_entries.append((name, static_summary))
        write_event(
            call_id,
            depth,
            phase,
            function_name,
            source_file,
            summarized_entries,
            kind="value",
        )

    if callback_values:
        try:
            jax.debug.callback(callback, *callback_values, ordered=True)
        except TypeError:
            jax.debug.callback(callback, *callback_values)
    else:
        callback()


def bind_arguments(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(fn)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def run_with_local_capture(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    captured_locals: dict[str, Any] = {}
    previous_trace = sys.gettrace()
    target_code = fn.__code__

    def local_tracer(frame, event, arg):
        if event == "return":
            captured_locals.update(frame.f_locals)
        return local_tracer

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is target_code:
            return local_tracer
        return previous_trace(frame, event, arg) if previous_trace is not None else None

    sys.settrace(tracer)
    try:
        result = fn(*args, **kwargs)
    finally:
        sys.settrace(previous_trace)

    return result, captured_locals


def return_variables(
    variables: list[str],
    arguments: dict[str, Any],
    local_variables: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return_names = [
        name for name in variables if name not in arguments and name not in local_variables
    ]
    if isinstance(result, tuple):
        return {
            name: result[index]
            for index, name in enumerate(return_names)
            if index < len(result)
        }

    if len(return_names) == 1:
        return {return_names[0]: result}

    return {}


def make_debug_wrapper(function_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        call_id = next(DUMP_CALL_COUNTER)
        stack = current_stack()
        depth = len(stack)
        stack.append(call_id)

        variables = TARGET_VARIABLES_BY_FUNCTION[function_name]
        arguments = bind_arguments(fn, args, kwargs)
        source_file = inspect.getsourcefile(fn)

        call_entries = []
        for name in variables:
            if name in arguments:
                call_entries.append((name, arguments[name]))

        write_event(
            call_id,
            depth,
            "call",
            function_name,
            source_file,
            [(name, summarize(value)) for name, value in call_entries],
            kind="trace",
        )

        try:
            result, local_variables = run_with_local_capture(fn, args, kwargs)
        except Exception as exc:
            dump_event(
                call_id,
                depth,
                "return",
                function_name,
                source_file,
                [("exception", repr(exc))],
            )
            raise
        finally:
            stack.pop()

        returns = return_variables(variables, arguments, local_variables, result)
        return_entries = []
        for name in variables:
            if name in local_variables and name not in arguments:
                return_entries.append((name, local_variables[name]))
            elif name in returns:
                return_entries.append((name, returns[name]))
            elif name not in arguments:
                return_entries.append((name, "<unavailable: not an argument or return value>"))

        write_event(
            call_id,
            depth,
            "return",
            function_name,
            source_file,
            [(name, summarize(value)) for name, value in return_entries],
            kind="trace",
        )
        dump_event(
            call_id,
            depth,
            "return",
            function_name,
            source_file,
            call_entries + return_entries,
        )
        return result

    return wrapper


def patch_targets() -> None:
    sys.path.insert(0, str(PINN_POISSON_DIR))

    import pinn

    for function_name in TARGET_VARIABLES_BY_FUNCTION:
        original = getattr(pinn, function_name, None)
        if original is None:
            print(f"[SETUP] missing pinn.{function_name}; skipping", flush=True)
            continue

        setattr(pinn, function_name, make_debug_wrapper(function_name, original))
        print(f"[SETUP] patched pinn.{function_name}", flush=True)


def run_workload() -> None:
    print(f"[SETUP] running workload: {PINNPY_FILE}", flush=True)
    runpy.run_path(str(PINNPY_FILE), run_name="__main__")


def main() -> None:
    DUMP_FILE.write_text("")
    append_dump(
        "# [trace] preserves Python/JAX tracing call nesting; [value] is emitted by "
        "jax.debug.callback at runtime and groups call arguments with return values."
    )
    print(f"[SETUP] writing debug dump to: {DUMP_FILE}", flush=True)
    patch_targets()
    run_workload()


if __name__ == "__main__":
    main()
