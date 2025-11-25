import asyncio # 비동기 처리를 위함
from fastapi import FastAPI, File, UploadFile, Depends, Form, HTTPException
from fastapi.staticfiles import StaticFiles # 파일 서버 임포트
from fastapi.middleware.cors import CORSMiddleware # CORS 설정
from sqlmodel import SQLModel, Session, select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel # API 입력 모델 임포트
import uuid # 파일의 고유한 이름을 만들기
from database import async_engine, create_db_and_tables, get_async_session
import models
import security
import shutil # 파일 저장을 위한 shutil 라이브러리 추가
# shutil 모듈은 파일 및 디렉토리 복사, 이동, 삭제 등 파일 시스템 관리 작업을 위한 고수준 함수들을 제공하는 표준 라이브러리입니다.

import os # uploads 폴더 생성을 위해 추가
# 운영체제에서 제공하는 기능을 파이썬 프로그램에서 쉽게 사용할 수 있도록 해주는 기본 모듈

import sys

import cv2

import numpy as np # 행렬 연산을 위해 추가
import requests # 모델 파일 다운로드를 위해 추가
import bz2 # 압축 해제를 위한 라이브러리
import traceback # 구체적인 에러 위치를 찾기 위해 추가

# 윈도우 호환성 패치 (PosixPath 에러 방지)
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from ultralytics import YOLO
#===============================================================================================================



# -- uploads 폴더 설정 --
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# DLL 충돌 방지 및 경로 설정
if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(os.getcwd())
    except Exception:
        pass
os.environ['PATH'] = os.getcwd() + ';' + os.environ['PATH']

# FastAPI 먼저 선언
app = FastAPI()

# (중요) React(3000번)에서 오는 요청을 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # React 서버 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    await create_db_and_tables()

# /get-analysis/ API가 받을 JSON 요청 본문 모델
class AnalysisLogin(BaseModel):
    request_id: int
    password: str

# =======================================================================

# AI 모델 파일 자동 다운로드 함수 (YOLOv8 Face 모델 및 번호판(license-plate) 모델)
def check_and_download_files():
    base_path = os.getcwd()
    
    # 1. YOLOv8 Face 모델 (얼굴 인식 전용 모델)
    face_model_name = "yolov8n-face.pt"
    face_model_path = os.path.join(base_path, face_model_name)

    # 2. 번호판 인식 모델
    plate_model_name = "yolov8n-license-plate.pt"
    plate_model_path = os.path.join(base_path, plate_model_name)

    # 3. 코덱 DLL (영상 저장용)
    target_dll = "openh264-1.8.0-win64.dll"
    dll_path = os.path.join(base_path, target_dll)

    # 번호판 모델이 너무 작으면(잘못된 다운로드) 삭제
    if os.path.exists(plate_model_path):
        if os.path.getsize(plate_model_path) < 5 * 1024 * 1024: # 1MB 이하면 가짜 파일로 간주
            print("잘못된 번호판 모델입니다. 삭제 후 다시 다운로드합니다.")
            try: os.remove(plate_model_path)
            except: pass

    # DLL 청소
    if os.path.exists(dll_path):
        size = os.path.getsize(dll_path)
        if size < 500000 or size > 900000:
            try: os.remove(dll_path)
            except: pass
    
    for f in os.listdir(base_path):
        if f.startswith("openh264") and f.endswith(".dll") and f != target_dll:
            try: os.remove(os.path.join(base_path, f))
            except: pass

# =======================================================================

    # YOLO Face(얼굴 인식) 모델 다운로드
    if not os.path.exists(face_model_path):
        print(f"얼굴 인식 모델 다운로드 중... ({face_model_name})")
        try:
            # 얼굴 인식 특화 YOLO 모델
            url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
            r = requests.get(url, stream=True)
            with open(face_model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except: pass

# =======================================================================

    # 번호판(license-plate) 모델 다운로드
    if not os.path.exists(plate_model_path):
        print(f"번호판 모델 다운로드 중... ({plate_model_name})")
        try:
            # 1순위 오픈소스 번호판 모델 링크 (GitHub Raw)
            url = "https://raw.githubusercontent.com/ablanco1950/LicensePlate_Yolov8_MaxFilters/main/best.pt"
            r = requests.get(url, stream=True)

            if r.status_code != 200:
                with open(plate_model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(" -> 번호판 모델 다운로드 완료")
            else:
                print(f"❌ 1차 다운로드 실패 (Status: {r.status_code}). 2차 시도...")
                # 2순위 링크 (백업)
                url2 = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
                r2 = requests.get(url2, stream=True)
                if r2.status_code == 200:
                    with open(plate_model_path, 'wb') as f:
                        for chunk in r2.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(" -> 번호판 모델(백업) 다운로드 완료")
                else:
                    print("❌ 모든 다운로드 링크 실패. 인터넷 상태를 확인하거나 수동으로 파일을 넣어주세요.")
        except Exception as e:
            print(f"번호판 모델 다운로드 에러: {e}")

# =======================================================================

    # 코덱 DLL 다운로드 (v1.8.0)
    if not os.path.exists(dll_path):
        print(f"코덱 DLL 다운로드 중... ({target_dll})")
        try:
            url = "http://ciscobinary.openh264.org/openh264-1.8.0-win64.dll.bz2"
            r = requests.get(url, stream=True)
            decompressed_data = bz2.decompress(r.content)
            with open(dll_path, 'wb') as f: f.write(decompressed_data)
            print(" -> DLL 설치 완료")
        except Exception as e:
            print(f" -> DLL 실패: {e}")

    return face_model_path, plate_model_path

# -----------------------------------------------

# 얼굴 = 타원, 번호판 = 네모로 블러 처리 및 AI 분석
def process_video_for_privacy(video_path: str, original_filename: str) -> dict:

    try:
        # 파일 체크 및 다운로드
        face_model_path, plate_model_path = check_and_download_files()
        
        print(f"'{video_path}' YOLOv8 얼굴 + 번호판 분석 시작.....")
        
        # 모델 로드 시도 (에러 발생 시 터미널에 출력)
        if not os.path.exists(plate_model_path):
             print("❌ 번호판 모델 파일이 없습니다! 다운로드에 실패했습니다.")
             return {"error": "번호판 모델 파일이 없습니다. 서버 로그를 확인해주세요."}

        
        # 두 개의 모델을 로드
        face_model = YOLO(face_model_path)
        plate_model = YOLO(plate_model_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "비디오 파일 열기 실패"}
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0.0: fps = 30.0

        unique_id = str(uuid.uuid4())
        blurred_filename = f"blur_{original_filename}"
        blurred_filepath = os.path.join(UPLOAD_DIRECTORY, blurred_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(blurred_filepath, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print("avc1 실패, mp4v로 전환")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(blurred_filepath, fourcc, fps, (frame_width, frame_height))

        total_detections = 0

        frame_count = 0

        # 번호판 검증 함수
        def is_valid_plate(x1, y1, x2, y2, frame_w, frame_h):
            w = x2 - x1
            h = y2 - y1
            
            # 1. 비율 검사 : 가로가 세로보다 훨씬 길어야 함 (1.5~6배)
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 6.0:
                return False
            
            # 2. 크기 검사 : 화면 전체의 60%를 넘는 거대한 물체는 번호판 아님
            plate_area = w * h
            frame_area = frame_w * frame_h
            if plate_area / frame_area > 0.6:
                return False
            
            # 3. 최소 크기 : 너무 작은 점 같은 건 무시
            if w < 30 or h < 10:
                return False
            
            return True

        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # 진행 상황 로깅 
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...", end='\r')

            # 1. 얼굴 인식 (타원형 블러) + CCTV 최적화 (imgsz=1280)
            # imgsz=1280: 입력 이미지를 크게 키워서 분석하므로 멀리 있는 얼굴도 잡힘
            # conf : 민감도 -> conf=(이 값 이상인 것만 잡음)
            face_results = face_model.predict(frame, conf=0.20, imgsz=1280, verbose=False)

            # 탐지된 결과 루프
            if face_results:
                for result in face_results:
                    if result is None or not hasattr(result, 'boxes') or result.boxes is None: continue

                    for box in result.boxes:
                        total_detections += 1
                        
                        # 좌표 추출
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # 얼굴 패딩
                        w, h = x2 - x1, y2 - y1
                        pad_x = int(w * 0.1)
                        pad_y_top = int(h * 0.2)
                        pad_y_bot = int(h * 0.2)
                        
                        bx1 = max(0, x1 - pad_x)
                        by1 = max(0, y1 - pad_y_top)
                        bx2 = min(frame_width, x2 + pad_x)
                        by2 = min(frame_height, y2 + pad_y_bot)
                        
                        roi = frame[by1:by2, bx1:bx2]
                        if roi.size == 0: continue
                        
                        try:
                            kw = int((bx2-bx1)/1.5) | 1
                            kh = int((by2-by1)/1.5) | 1
                            blurred = cv2.GaussianBlur(roi, (kw, kh), 0)
                            
                            mask = np.zeros_like(roi)
                            cv2.ellipse(mask, ((bx2-bx1)//2, (by2-by1)//2), ((bx2-bx1)//2, (by2-by1)//2), 0, 0, 360, (255, 255, 255), -1)
                            frame[by1:by2, bx1:bx2] = np.where(mask > 0, blurred, roi)
                        except: pass

            # 2. 번호판 인식 (직사각형 블러)
            # 민감도 0.25
            plate_results = plate_model.predict(frame, conf=0.25, imgsz=1280, verbose=False)


            if plate_results:
                for result in plate_results:
                    if result is None or not hasattr(result, 'boxes') or result.boxes is None: continue

                    for box in result.boxes:
                        total_detections += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        if is_valid_plate(x1, y1, x2, y2, frame_width, frame_height):
                            total_detections += 1
                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0: continue

                            try:
                                # 직사각형 블러
                                kw = int((x2-x1)/2) | 1
                                kh = int((y2-y1)/2) | 1
                                blurred_plate = cv2.GaussianBlur(roi, (kw, kh), 0)
                                frame[y1:y2, x1:x2] = blurred_plate
                            except: pass

            out.write(frame)

        cap.release()
        out.release()

        print(f"'{blurred_filepath}' YOLO 분석 완료. (총 탐지: {total_detections})")
        
        return {
            "detection_summary": {"faces_blurred": total_detections},
            "analyzed_video_url": f"/static/{blurred_filename}"
        }

    except Exception as e:
        # [핵심] 에러 발생 시 여기서 터미널에 상세 내용을 뿌립니다.
        print("❌ 분석 중 치명적인 오류 발생:")
        traceback.print_exc() # <--- 이게 터미널에 빨간 글씨로 뜹니다.
        return {"error": f"서버 내부 오류: {str(e)}"}

# -----------------------------------------------------

# 파일 서버 마운트
app.mount("/static", StaticFiles(directory=UPLOAD_DIRECTORY), name="static")

@app.post("/request-analysis/")
async def upload_video(
    video: UploadFile = File(...),
    email: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_async_session)
):
    # 1. 원본 영상 저장
    original_filename = video.filename
    original_filepath = os.path.join(UPLOAD_DIRECTORY, original_filename)
    
    with open(original_filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # 2. AI 분석 실행
    ai_result_dict = process_video_for_privacy(original_filepath, original_filename)

    if "error" in ai_result_dict:
        raise HTTPException(status_code=500, detail=ai_result_dict["error"])

    # 3. DB 저장
    hashed_pw = security.get_password_hash(password)
    new_request = models.AnalysisRequest(
        email=email,
        password_hash=hashed_pw,
        original_video_filename=video.filename,
        original_video_path=original_filepath,
        analyzed_video_path=ai_result_dict["analyzed_video_url"],
        status="COMPLETED"
    )

    session.add(new_request)
    try:
        await session.commit()
        await session.refresh(new_request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 저장 실패: {str(e)}")

    return {
        "message": "분석 완료",
        "request_id": new_request.id,
        "filename": new_request.original_video_filename,
        "analysis": ai_result_dict["detection_summary"],
        "analyzed_video_url": ai_result_dict["analyzed_video_url"]
    }

# ID와 비밀번호로 분석 결과 가져오기 API
@app.post("/get-analysis/")
async def get_analysis(
    login_data: AnalysisLogin,
    session: AsyncSession = Depends(get_async_session)
):
    # 2. DB에서 ID로 게시글 찾기
    statement = select(models.AnalysisRequest).where(
        models.AnalysisRequest.id == login_data.request_id
    )
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()

    # 3. 게시글이 없는 경우
    if not db_post:
        raise HTTPException(status_code=404, detail="해당 ID의 분석 요청을 찾을 수 없습니다.")

    if not security.verify_password(login_data.password, db_post.password_hash):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")

    # 성공: 게시글 전체 정보를 React에게 반환 (DB에 저장된 그대로)
    return db_post