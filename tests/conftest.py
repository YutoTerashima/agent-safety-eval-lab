from pathlib import Path
import os


def pytest_configure(config):
    os.chdir(Path(__file__).resolve().parents[1])
