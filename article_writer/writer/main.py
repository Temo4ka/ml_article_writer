def main():
    from .gui import run_app
    run_app()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Чтобы работало и как скрипт (python main.py), и как модуль (python -m article_writer.writer.main)
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from article_writer.writer.gui import run_app
    except ImportError:
        from .gui import run_app
    run_app()
