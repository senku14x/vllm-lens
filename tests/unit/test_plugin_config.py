"""CPU tests for ``_patched_create_engine_config`` environment defaults."""

from types import SimpleNamespace

import pytest
import torch

import vllm_lens._activations_plugin as plugin


def _fake_config(dtype):
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=dtype), use_v2_model_runner=False
    )


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(
        plugin, "_original_create_engine_config", lambda self, *a, **k: self._cfg
    )
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)

    def run(dtype):
        args = SimpleNamespace(
            worker_extension_cls=None, enforce_eager=False, _cfg=_fake_config(dtype)
        )
        plugin._patched_create_engine_config(args)
        return args

    return run


def test_fp32_model_defaults_triton_to_ieee(patched, monkeypatch):
    import os

    patched(torch.float32)
    assert os.environ["TRITON_F32_DEFAULT"] == "ieee"


def test_bf16_model_leaves_triton_alone(patched):
    import os

    patched(torch.bfloat16)
    assert "TRITON_F32_DEFAULT" not in os.environ


def test_explicit_triton_setting_wins(patched, monkeypatch):
    import os

    monkeypatch.setenv("TRITON_F32_DEFAULT", "tf32")
    patched(torch.float32)
    assert os.environ["TRITON_F32_DEFAULT"] == "tf32"


def test_cuda_graphs_off_forces_eager_and_worker_ext(patched, monkeypatch):
    monkeypatch.setenv("VLLM_LENS_CUDA_GRAPHS", "0")
    args = patched(torch.bfloat16)
    assert args.enforce_eager is True
    assert args.worker_extension_cls == plugin._WORKER_EXT


def test_default_keeps_cuda_graphs_but_not_compile(patched, monkeypatch):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    monkeypatch.delenv("VLLM_LENS_CUDA_GRAPHS", raising=False)
    args = patched(torch.bfloat16)
    assert args.enforce_eager is False
    assert args.worker_extension_cls == plugin._WORKER_EXT
    assert args.compilation_config.mode == CompilationMode.NONE
    assert args.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"),
        ("PIECEWISE", "FULL_DECODE_ONLY"),
        ("FULL", "FULL"),
        ("FULL_DECODE_ONLY", "FULL_DECODE_ONLY"),
    ],
)
def test_user_compilation_config_made_hook_compatible(
    monkeypatch, requested, expected
):
    pytest.importorskip("vllm")
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    monkeypatch.delenv("VLLM_LENS_CUDA_GRAPHS", raising=False)
    args = SimpleNamespace(
        enforce_eager=False,
        compilation_config={"mode": 3, "cudagraph_mode": requested},
    )
    assert plugin._configure_hook_compatible_graphs(args) is True
    assert args.compilation_config.mode == CompilationMode.NONE
    assert args.compilation_config.cudagraph_mode == CUDAGraphMode[expected]


def test_cudagraph_mode_none_means_eager(patched, monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.delenv("VLLM_LENS_CUDA_GRAPHS", raising=False)
    monkeypatch.setattr(
        plugin, "_original_create_engine_config", lambda self, *a, **k: self._cfg
    )
    args = SimpleNamespace(
        worker_extension_cls=None,
        enforce_eager=False,
        compilation_config={"cudagraph_mode": "NONE"},
        _cfg=_fake_config(torch.bfloat16),
    )
    plugin._patched_create_engine_config(args)
    assert args.enforce_eager is True


def test_explicit_enforce_eager_is_kept(monkeypatch):
    monkeypatch.delenv("VLLM_LENS_CUDA_GRAPHS", raising=False)
    args = SimpleNamespace(enforce_eager=True, compilation_config=None)
    assert plugin._configure_hook_compatible_graphs(args) is False
    assert args.enforce_eager is True


def test_compiled_config_is_rebuilt_eager(monkeypatch):
    """If vLLM keeps torch.compile on, the hooks could never fire: rebuild eager."""
    calls = []

    def fake_create(self, *a, **k):
        calls.append(self.enforce_eager)
        return SimpleNamespace(
            model_config=SimpleNamespace(
                dtype=torch.bfloat16, enforce_eager=self.enforce_eager
            ),
            compilation_config=SimpleNamespace(mode=0 if self.enforce_eager else 3),
            use_v2_model_runner=False,
        )

    monkeypatch.setattr(plugin, "_original_create_engine_config", fake_create)
    monkeypatch.setattr(plugin, "_configure_hook_compatible_graphs", lambda a: True)
    monkeypatch.delenv("VLLM_LENS_CUDA_GRAPHS", raising=False)
    args = SimpleNamespace(worker_extension_cls=None, enforce_eager=False)
    config = plugin._patched_create_engine_config(args)
    assert calls == [False, True]
    assert config.model_config.enforce_eager is True
