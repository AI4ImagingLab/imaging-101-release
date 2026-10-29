"""The optional generic backend owns its process and fails explicitly."""
import pytest
from src.linear_algebra import SourceBlas, current_backend


def test_failed_generic_library_startup_cleans_owned_worker(monkeypatch):
    monkeypatch.setenv('IMAGING_BLAS_LIBRARY', 'imaging_nonexistent_test_library.so')
    backend = SourceBlas()
    with pytest.raises(RuntimeError, match='generic BLAS IPC failed'):
        with backend:
            pytest.fail('invalid library was accepted')
    assert backend.process.poll() is not None
    assert current_backend() is None


def test_context_cleanup_on_callers_exception():
    backend = SourceBlas()
    with pytest.raises(ValueError, match='caller failure'):
        with backend:
            assert current_backend() is backend
            raise ValueError('caller failure')
    assert backend.process.returncode == 0
    assert current_backend() is None
