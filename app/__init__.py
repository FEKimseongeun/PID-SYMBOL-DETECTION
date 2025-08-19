# app/__init__.py
import os
from flask import Flask
from flask_cors import CORS


def create_app():
    # Flask 앱 인스턴스 생성
    app = Flask(__name__, template_folder='templates')
    CORS(app)

    # 애플리케이션 설정
    app.config['UPLOAD_FOLDER'] = os.path.abspath('temp_uploads')
    app.config['RESULT_FOLDER'] = os.path.abspath('temp_results')

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

    # 블루프린트(routes) 등록
    with app.app_context():
        from . import routes
        app.register_blueprint(routes.bp)

    return app