import os
import time
from app.services.detection_service import run_analysis_pipeline
from app import create_app  # Flask 앱 컨텍스트를 위해 import

# ------------------------------------------------------------------
# ✨ 1. 분석할 파일과 설정을 여기에 직접 지정합니다.
# ------------------------------------------------------------------
INPUT_PDF_PATH = "E:/2_EDE/BALTIC_3/01.HE_1_27.pdf"  # 👈 분석하고 싶은 PDF 파일 경로를 입력하세요.
OUTPUT_DIR_BASE = "test/temp_results"

# 웹 UI에서 입력하던 설정값들을 하드코딩
ANALYSIS_SETTINGS = {
    'targetLabels': ["26", "27", "28", "29", "31"],  # 탐지할 심볼 클래스
    'referenceText': "GCC-DLM-DDD-12600-00-0000-TH-PID-55104.",  # 기준 텍스트
    'referencePage': 0  # 기준 페이지
}


# ------------------------------------------------------------------

def simple_status_update(job_id, status, progress=0, message="", **kwargs):
    """
    분석 파이프라인에 전달할 간단한 상태 업데이트 함수입니다.
    웹소켓이나 DB 대신 간단히 콘솔에 출력만 합니다.
    """
    print(f"[진행률 {progress}%] {message}")


def main():
    """
    웹서버 없이 분석 파이프라인을 직접 실행하는 메인 함수입니다.
    """
    if not os.path.exists(INPUT_PDF_PATH):
        print(f"오류: 입력 파일 '{INPUT_PDF_PATH}'을(를) 찾을 수 없습니다.")
        return

    # 고유한 작업 ID 및 결과 폴더 생성
    job_id = f"local_run_{int(time.time())}"
    output_dir = os.path.join(OUTPUT_DIR_BASE, job_id)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print(f"🚀 로컬 분석을 시작합니다.")
    print(f" - 작업 ID: {job_id}")
    print(f" - 입력 파일: {INPUT_PDF_PATH}")
    print(f" - 결과 폴더: {output_dir}")
    print("=" * 50)

    start_time = time.time()

    # detection_service가 current_app.config를 사용하므로 앱 컨텍스트가 필요합니다.
    # 실제 서버를 띄우지 않고, 설정값만 가진 앱 컨텍스트를 만들어줍니다.
    app = create_app()
    with app.app_context():
        try:
            # ✨ 핵심: detection_service의 메인 함수를 직접 호출!
            result = run_analysis_pipeline(
                job_id=job_id,
                pdf_path=INPUT_PDF_PATH,
                settings=ANALYSIS_SETTINGS,
                output_dir=output_dir,
                update_job_status_func=simple_status_update,
                config_path="config/config.yaml"
            )

            if result.get('success'):
                print("\n" + "=" * 50)
                print("✅ 분석이 성공적으로 완료되었습니다!")
                print(f" - 총 처리 페이지: {result.get('total_pages', 'N/A')}")
                print(f" - 총 탐지 심볼: {result.get('total_symbols', 'N/A')}")
                print(f" - 엑셀 결과: {result.get('excel_path', 'N/A')}")
                print("=" * 50)
            else:
                print("\n❌ 분석에 실패했습니다.")

        except Exception as e:
            print(f"\n💥 치명적인 오류 발생: {e}")
            import traceback
            traceback.print_exc()  # 오류의 상세 내용을 출력

    end_time = time.time()
    print(f"\n⏱️ 총 소요 시간: {end_time - start_time:.2f}초")


if __name__ == '__main__':
    # 멀티프로세싱 시작 방식 설정 (detection_service와 동일하게)
    import multiprocessing

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()