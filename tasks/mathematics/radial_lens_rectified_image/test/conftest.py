"""Use a unique test temporary root when the caller did not select one.

Avoids a shared pytest-of-user directory with stale Windows ACLs. All generated
test submissions remain outside the task package.
"""
import tempfile
def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp=tempfile.mkdtemp(prefix='imaging101_lens_tests_')
