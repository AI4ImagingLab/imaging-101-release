"""Keep temporary submissions outside the task and avoid stale shared Windows ACLs."""
import tempfile

def pytest_configure(config):
    if config.option.basetemp is None:
        config.option.basetemp=tempfile.mkdtemp(prefix='imaging101_pnr_tests_')
