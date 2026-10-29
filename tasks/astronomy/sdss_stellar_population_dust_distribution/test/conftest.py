import tempfile
def pytest_configure(config):
    if config.option.basetemp is None:config.option.basetemp=tempfile.mkdtemp(prefix='imaging101_firefly_tests_')
