# run.py
from app import create_app
import multiprocessing

# Flask 앱 인스턴스 생성
app = create_app()

if __name__ == '__main__':
    # Flask 앱을 실행하기 전에 multiprocessing 시작 방식을 설정하는 것이 안전합니다.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 이미 설정된 경우 오류가 발생할 수 있으므로 무시합니다.

    print("🚀 P&ID 분석 웹 서버 시작...")
    print("http://127.0.0.1:5000 에서 접속 가능합니다.")
    app.run(debug=False, host='0.0.0.0', port=5000)