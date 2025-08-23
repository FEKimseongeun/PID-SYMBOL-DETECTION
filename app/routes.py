# app/routes.py
import os
import uuid
import threading
import zipfile
from datetime import datetime
from flask import (
    Blueprint, request, jsonify, send_file, render_template, current_app
)
from werkzeug.utils import secure_filename

# detection_service에서 메인 파이프라인 함수를 import
from .services.detection_service import run_analysis_pipeline

# Blueprint 객체 생성
bp = Blueprint('main', __name__)

# 작업 상태를 저장할 딕셔너리
analysis_jobs = {}


def update_job_status(job_id, status, progress=0, message="", error=None, results=None):
    job_info = {'status': status, 'progress': progress, 'message': message, 'error': error,
                'timestamp': datetime.now().isoformat()}
    if results: job_info['results'] = results
    analysis_jobs[job_id] = job_info


# --- ✨ [수정 1] 함수 시그니처에 'app' 추가 ---
def analyze_pdf_background(app, job_id, pdf_path, settings):
    """백그라운드에서 PDF 분석 실행"""
    # --- ✨ [수정 2] with app.app_context(): 로 전체 로직을 감싸기 ---
    with app.app_context():
        try:
            job_dir = os.path.join(current_app.config['RESULT_FOLDER'], job_id)
            result = run_analysis_pipeline(job_id, pdf_path, settings, job_dir, update_job_status)

            if result['success']:
                update_job_status(job_id, 'running', 95, '결과 파일 압축 중...')
                zip_path = os.path.join(job_dir, 'annotated_images.zip')
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, _, files in os.walk(result['annotated_images_dir']):
                        for file in files: zipf.write(os.path.join(root, file), file)
                final_results = {
                    'excelPath': result['excel_path'],
                    'zipPath': zip_path,
                    'totalPages': result['total_pages'],  # 추가
                    'totalSymbols': result['total_symbols']  # 추가
                }
                update_job_status(job_id, 'completed', 100, '분석 완료!', results=final_results)
            else:
                raise Exception("PDF 분석 실패")
        except Exception as e:
            print(f"백그라운드 작업 오류: {e}")
            update_job_status(job_id, 'error', 0, '오류가 발생했습니다.', str(e))


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error': '파일이 없습니다'}), 400
    file = request.files['file']
    if not file.filename.lower().endswith('.pdf'): return jsonify({'error': 'PDF 파일만 업로드 가능합니다'}), 400
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return jsonify({'fileId': file_id, 'filename': file.filename, 'message': '파일 업로드 성공'})


@bp.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if 'fileId' not in data or 'settings' not in data: return jsonify({'error': '요청 데이터가 올바르지 않습니다'}), 400
    file_id = data['fileId']
    uploaded_files = [f for f in os.listdir(current_app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
    if not uploaded_files: return jsonify({'error': '업로드된 파일을 찾을 수 없습니다'}), 404
    pdf_path = os.path.join(current_app.config['UPLOAD_FOLDER'], uploaded_files[0])
    job_id = str(uuid.uuid4())

    # --- ✨ [수정 3] 실제 app 객체를 가져와 스레드에 전달 ---
    app = current_app._get_current_object()
    thread = threading.Thread(target=analyze_pdf_background, args=(app, job_id, pdf_path, data['settings']))
    thread.start()

    return jsonify({'jobId': job_id, 'status': 'started', 'message': '분석이 시작되었습니다'})


@bp.route('/api/status/<job_id>')
def get_status(job_id):
    job = analysis_jobs.get(job_id)
    return jsonify(job) if job else (jsonify({'error': '작업을 찾을 수 없습니다'}), 404)


@bp.route('/api/download/excel/<job_id>')
def download_excel(job_id):
    job = analysis_jobs.get(job_id, {})
    if job.get('status') != 'completed': return jsonify({'error': '작업이 완료되지 않았습니다'}), 400
    excel_path = job.get('results', {}).get('excelPath')
    if not excel_path or not os.path.exists(excel_path): return jsonify({'error': '결과 파일을 찾을 수 없습니다'}), 404
    return send_file(excel_path, as_attachment=True)


@bp.route('/api/download/images/<job_id>')
def download_images(job_id):
    job = analysis_jobs.get(job_id, {})
    if job.get('status') != 'completed': return jsonify({'error': '작업이 완료되지 않았습니다'}), 400
    zip_path = job.get('results', {}).get('zipPath')
    if not zip_path or not os.path.exists(zip_path): return jsonify({'error': '이미지 파일을 찾을 수 없습니다'}), 404
    return send_file(zip_path, as_attachment=True)