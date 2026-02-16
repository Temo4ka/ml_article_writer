"""
Веб-интерфейс для генерации статей (Flask).
Открывается в браузере на Mac/Windows, без зависимости от Tk.
"""
import io
from pathlib import Path

from flask import Flask, request, jsonify, send_file, render_template
from docx import Document
from docx.shared import Pt

from .model_loader import load_model, generate_article

# Глобальное состояние модели (один процесс)
_model = None
_tokenizer = None

def _get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = load_model()
    return _model, _tokenizer


def _save_docx(text: str) -> io.BytesIO:
    buf = io.BytesIO()
    doc = Document()
    for para in text.strip().split("\n\n"):
        p = doc.add_paragraph(para)
        p.paragraph_format.space_after = Pt(12)
    doc.save(buf)
    buf.seek(0)
    return buf


def create_app():
    app = Flask(__name__, template_folder=Path(__file__).resolve().parent / "templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/load_model", methods=["POST"])
    def api_load_model():
        try:
            _get_model()
            return jsonify({"status": "ok"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        try:
            model, tokenizer = _get_model()
        except Exception as e:
            return jsonify({"error": f"Модель не загружена: {e}"}), 400
        data = request.get_json() or {}
        topic = (data.get("topic") or "").strip()
        word_count = max(100, min(5000, int(data.get("word_count") or 500)))
        if not topic:
            return jsonify({"error": "Укажите тему статьи"}), 400
        try:
            text = generate_article(model, tokenizer, topic=topic, word_count=word_count)
            return jsonify({"text": text})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/export_docx", methods=["POST"])
    def api_export_docx():
        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "Нет текста"}), 400
        try:
            buf = _save_docx(text)
            return send_file(
                buf,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                as_attachment=True,
                download_name="article.docx",
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def run_app(host: str = "127.0.0.1", port: int = 5000, open_browser: bool = False):
    app = create_app()
    url = f"http://{host}:{port}/"
    if open_browser:
        try:
            import webbrowser
            from threading import Timer
            def _open():
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            Timer(1.0, _open).start()
        except Exception:
            pass
    print(f"Откройте в браузере: {url}")
    app.run(host=host, port=port, debug=False, threaded=True)
