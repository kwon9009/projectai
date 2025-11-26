import asyncio # 비동기 처리를 위함 (동시 작업 처리를 위해)
from fastapi import FastAPI, File, UploadFile, Depends, Form, HTTPException
# API : 서비스의 요청과 응답을 처리하는 기능
# FastAPI : 파이썬 웹 프레임 워크
# File, UploadFile : 파일 업로드를 처리하는 도구
# Depends : 의존성 주입 (DB 세션 등을 함수에 전달할 때 사용)
# Form : HTML 폼 데이터(이메일, 비밀번호 등)를 받기 위함
# HTTPException : 에러 발생 시 사용자에게 적절한 에러 코드를 보내기 위함

from fastapi.staticfiles import StaticFiles # 정적 파일(이미지, 영상 등)을 웹에서 접근 가능하게 하는 도구
from fastapi.middleware.cors import CORSMiddleware # 다른 도메인(React 등)에서 서버로 요청을 보낼 수 있게 허용하는 보안 설정 도구
from sqlmodel import SQLModel, Session, select # DB 모델 정의 및 쿼리 작성을 위한 도구
from sqlalchemy.ext.asyncio import AsyncSession # 비동기 DB 처리를 위한 세션 도구
from pydantic import BaseModel # 데이터 검증을 위한 모델 도구 (입력 데이터 형식 정의)
import uuid # 파일명 중복 방지를 위한 고유 ID 생성 도구
from database import async_engine, create_db_and_tables, get_async_session # database.py에서 정의한 DB 연결 도구들 가져오기
import models # models.py에서 정의한 DB 테이블 구조 가져오기
import security # security.py에서 정의한 비밀번호 암호화 도구 가져오기
import shutil # 파일 저장(복사, 이동)을 위한 파일 관리 도구 라이브러리
import os # 파일 경로, 폴더 생성 등 운영체제 기능 사용 도구
import sys # 시스템 관련 기능을 사용하기 위한 모듈
import cv2  # OpenCV: 영상 처리 핵심 라이브러리
import numpy as np # 행렬 연산 도구 (이미지 데이터 처리)
import requests # 인터넷에서 파일(AI 모델 등)을 다운로드하기 위한 도구
import bz2 # 압축 해제를 위한 라이브러리
import traceback # 에러 발생 시 자세한 원인을 출력하기 위한 도구

from ultralytics import YOLO # yolov8 라이브러리

# # 윈도우 호환성 패치 (PosixPath 에러 방지)
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


#===============================================================================================================

# -- uploads 폴더 설정 --
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
# 'uploads'라는 이름의 폴더가 없으면 새로 만듦.

# FastAPI 앱 생성
app = FastAPI()

#===============================================================================================================

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # React 프론트엔드 주소만 허용
    allow_credentials=True, # 쿠키 등 인증 정보 허용
    allow_methods=["*"], # 모든 HTTP 메서드(GET, POST) 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# 서버가 켜질 때 딱 한 번 실행되어 DB 테이블을 만듦
@app.on_event("startup")
async def on_startup():
    await create_db_and_tables()

#===============================================================================================================

# /get-analysis/ API가 받을 JSON 요청 본문 모델
# 결과 조회 시 클라이언트가 보내야 할 데이터 형식(ID는 숫자, 비번은 문자열)을 정의
class AnalysisLogin(BaseModel):
    request_id: int
    password: str

#===============================================================================================================

# AI 모델 파일 자동 다운로드 함수 (YOLOv8 Face 모델 및 번호판(license-plate) 모델)
def check_and_download_files():
    base_path = os.getcwd() # 현재 작업 경로
    
    # 필요한 파일들의 경로 설정
    # 1. YOLOv8 Face 모델 (얼굴 인식 전용 모델)
    face_model_name = "yolov8n-face.pt"
    face_model_path = os.path.join(base_path, face_model_name)

    # 2. 번호판 인식 모델
    plate_model_name = "yolov8n-license-plate.pt"
    plate_model_path = os.path.join(base_path, plate_model_name)

    # 3. 코덱 DLL (영상 저장용)
    target_dll = "openh264-1.8.0-win64.dll"
    dll_path = os.path.join(base_path, target_dll)

    # 불량 파일 검사 및 삭제 (파일 크기가 너무 작으면 제대로 다운로드 안 된 것으로 간주)
    if os.path.exists(plate_model_path):
        if os.path.getsize(plate_model_path) < 5 * 1024 * 1024: # 5MB 미만이면 삭제
            print("잘못된 번호판 인식 모델입니다. 삭제 후 다시 다운로드합니다.")
            try: os.remove(plate_model_path)
            except: pass

    # DLL 청소
    if os.path.exists(dll_path):
        if os.path.getsize(dll_path) < 500000: # 500KB 미만이면 삭제
            try: os.remove(dll_path)
            except: pass
    
    # 잘못된 버전의 DLL 파일 정리 (충돌 방지)
    for f in os.listdir(base_path):
        if f.startswith("openh264") and f.endswith(".dll") and f != target_dll:
            try: os.remove(os.path.join(base_path, f))
            except: pass

#===============================================================================================================

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

#===============================================================================================================

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

#===============================================================================================================

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

    return face_model_path, plate_model_path # 모델 파일 경로 2개를 반환

#===============================================================================================================

# 얼굴 = 타원, 번호판 = 네모로 블러 처리 및 AI 분석
def process_video_for_privacy(video_path: str, original_filename: str) -> dict:

    try:
        # 파일 체크 및 다운로드
        face_model_path, plate_model_path = check_and_download_files()
        
        print(f"'{video_path}' YOLOv8 얼굴 + 번호판 분석 시작.....")
        
        # 모델 로드 시도 (에러 발생 시 터미널에 출력)
        if not os.path.exists(plate_model_path):
             print("번호판 모델 파일이 없습니다! 다운로드에 실패했습니다.")
             return {"error": "번호판 모델 파일이 없습니다. 서버 로그를 확인해주세요."}

        
        # 두 개의 모델을 로드 (메모리에 올림)
        face_model = YOLO(face_model_path)
        plate_model = YOLO(plate_model_path)

        # 영상 파일 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "비디오 파일 열기 실패"}
        
        # 영상 정보 가져오기 (너비, 높이, FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0.0: fps = 30.0 # FPS 정보 없으면 30으로 가정

        # 결과 파일명 및 경로 설정
        blurred_filename = f"blur_{original_filename}" # 결과 영상은 'blur_' 접두어 붙임
        blurred_filepath = os.path.join(UPLOAD_DIRECTORY, blurred_filename)
        
        # 영상 저장 설정 (코덱)
        # avc1 (웹 호환성이 좋음)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(blurred_filepath, fourcc, fps, (frame_width, frame_height))

        # avc1 실패 시 mp4v로 전환
        if not out.isOpened():
            print("avc1 실패, mp4v로 전환")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(blurred_filepath, fourcc, fps, (frame_width, frame_height))

        total_detections = 0 # 총 탐지 횟수 카운트

        frame_count = 0 # 현재 처리 중인 프레임 번호

        # 번호판 기억 저장소
        plate_memory = {}

        # 번호판 검증 함수
        def is_valid_plate(x1, y1, x2, y2, frame_w, frame_h):
            # 너비, 높이 계산
            w = x2 - x1
            h = y2 - y1
            
            # [추가] 높이가 0이거나 이상하면 바로 무시 (ZeroDivisionError 방지)
            if h <= 0: 
                return False
            
            # 거리가 멀리 떨어진 번호판(너비 50px 미만) 탐지
            if w < 50:
                return True

            # 1. 비율 검사 : 가로가 세로보다 적당히 길어야 함
            aspect_ratio = w / h
            if aspect_ratio < 1.0 or aspect_ratio > 8.0:
                return False
            
            # 2. 크기 검사 : 화면 전체의 3%를 넘는 거대한 물체는 번호판 아님
            plate_area = w * h
            frame_area = frame_w * frame_h
            if plate_area / frame_area > 0.03:
                return False
            
            return True

        # 프레임 반복 처리 시작
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break # 영상 끝나면 종료

            # 진행 상황 로그 출력 (30프레임마다)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...", end='\r')

            # 1. 얼굴 인식 (타원형 블러) + CCTV 최적화 (imgsz=1920)
            # imgsz=1920: 분석 해상도를 키워서 분석하므로 멀리 있는 얼굴도 잡게 함.
            # conf : 민감도 -> conf=(이 값 이상인 것만 잡음)
            # track : 단순히 찾기만 하는게 아닌, 물체의 이동 경로를 계산함.
            # persist = True (기억 유지) : 이전 장면의 정보를 계속 기억함. 객체가 사라졌다가 나타날 때 필요
            # tracker = "bytetrack.yaml" : 흐릿하거나 신뢰도가 낮을 물체도 연결해주는 알고리즘
            face_results = face_model.track(frame, conf=0.25, imgsz=1920, augment=True, persist=True, tracker="bytetrack.yaml", verbose=False)

            # 탐지된 결과 루프
            if face_results:
                for result in face_results:
                    # 결과 유효성 검사
                    if result is None or not hasattr(result, 'boxes') or result.boxes is None: continue

                    for box in result.boxes:
                        total_detections += 1
                        
                        # 좌표 추출
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # 얼굴이 너무 크면 오탐지로 무시하기 (번호판 인식 할 때 얼굴로 인식되는 경우 있음)
                        face_w = x2 - x1
                        face_h = y2 - y1
                        if (face_w * face_h) > (frame_width * frame_height * 0.10): # 화면의 10% 이상이면 오탐지
                            continue # 밑에 블러 코드 실행하지 않고 건너뜀

                        # 얼굴 비율 검사 추가 (가로로 길거나, 세로로 얇은 건 얼굴이 아님)
                        face_aspect_ratio = face_w / face_h
                        if face_aspect_ratio > 2.0 or face_aspect_ratio < 0.25:
                            continue

                        # 블러 영역 설정 (얼굴보다 조금 더 넓게 잡기)
                        w, h = x2 - x1, y2 - y1
                        pad_x = int(w * 0.1)
                        pad_y_top = int(h * 0.2)
                        pad_y_bot = int(h * 0.2)
                        
                        # 좌표 보정 (화면 밖으로 나가지 않게)
                        bx1 = max(0, x1 - pad_x)
                        by1 = max(0, y1 - pad_y_top)
                        bx2 = min(frame_width, x2 + pad_x)
                        by2 = min(frame_height, y2 + pad_y_bot)
                        
                        roi = frame[by1:by2, bx1:bx2] # 얼굴 영역 자르기
                        if roi.size == 0: continue
                        
                        try:
                            # 블러 강도 설정
                            kw = int((bx2-bx1)/1.5) | 1
                            kh = int((by2-by1)/1.5) | 1
                            blurred = cv2.GaussianBlur(roi, (kw, kh), 0)
                            
                            # 타원형 블러 만들기
                            mask = np.zeros_like(roi)
                            cv2.ellipse(mask, ((bx2-bx1)//2, (by2-by1)//2), ((bx2-bx1)//2, (by2-by1)//2), 0, 0, 360, (255, 255, 255), -1)

                            # 원본 이미지에 타원형 블러 합성
                            frame[by1:by2, bx1:bx2] = np.where(mask > 0, blurred, roi)
                        except: pass

            # 2. 번호판 인식 (직사각형 블러)
            # 민감도 0.05
            # augment=True : 이미지를 여러 번 변형해서 꼼꼼하게 검사
            plate_results = plate_model.track(frame, conf=0.05, imgsz=1920, augment=True, persist=True, tracker="bytetrack.yaml", verbose=False)

            current_frame_ids = [] # 이번 프레임에서 잡은 번호판 ID들

            if plate_results:
                for result in plate_results:
                    # 결과 유효성 검사
                    if result is None or not hasattr(result, 'boxes') or result.boxes is None: continue

                    for box in result.boxes:
                        
                        # 트래킹 ID 가져오기
                        track_id = int(box.id.item() if box.id is not None else -1)

                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # 번호판 검증 통과한 번호판만 처리
                        if is_valid_plate(x1, y1, x2, y2, frame_width, frame_height):
                            total_detections += 1

                            # 1. 블러 처리
                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0: continue

                            try:
                                kw = int((x2-x1)/2) | 1
                                kh = int((y2-y1)/2) | 1
                                blurred_plate = cv2.GaussianBlur(roi, (kw, kh), 0)
                                frame[y1:y2, x1:x2] = blurred_plate
                            except: pass

                            # 2. 메모리에 저장
                            if track_id != -1:
                                plate_memory[track_id] = {'coords': (x1, y1, x2, y2), 'life': 15}
                                current_frame_ids.append(track_id)

            # 놓친 번호판 블러 처리
            keys_to_remove = []
            for tid, info in plate_memory.items():
                if tid not in current_frame_ids: # 방금 놓쳤다면
                    # 기억된 좌표로 블러
                    lx1, ly1, lx2, ly2 = info['coords']

                    # 화면 밖 체크
                    lx1, ly1 = max(0, lx1), max(0, ly1)
                    lx2, ly2 = min(frame_width, lx2), min(frame_height, ly2)

                    roi = frame[ly1:ly2, lx1:lx2]
                    if roi.size > 0:
                        try:
                            # 블러 정도
                            kw = int((lx2-lx1)/2) | 1
                            kh = int((ly2-ly1)/2) | 1
                            blurred_plate = cv2.GaussianBlur(roi, (kw, kh), 0)
                            frame[ly1:ly2, lx1:lx2] = blurred_plate
                        except: pass
                    
                    # 수명 감소 (0이 되면 삭제)
                    info['life'] -= 1
                    if info['life'] <= 0:
                        keys_to_remove.append(tid)

            # 수명 다한 기억 삭제
            for k in keys_to_remove:
                del plate_memory[k]

            # 처리된 프레임 저장
            out.write(frame)

        cap.release()
        out.release()

        print(f"'{blurred_filepath}' YOLO 분석 완료. (총 탐지: {total_detections})")
        
        # 결과 반환
        return {
            "detection_summary": {"faces_blurred": total_detections},
            "analyzed_video_url": f"/static/{blurred_filename}"
        }

    except Exception as e:
        # [핵심] 에러 발생 시 여기서 터미널에 상세 내용을 뿌립니다.
        print("❌ 분석 중 치명적인 오류 발생:")
        traceback.print_exc() # <--- 이게 터미널에 빨간 글씨로 뜹니다.
        return {"error": f"서버 내부 오류: {str(e)}"}

#===============================================================================================================

# 정적 파일(영상 등) 설정
app.mount("/static", StaticFiles(directory=UPLOAD_DIRECTORY), name="static")

# 1. 분석 요청 API
@app.post("/request-analysis/")
async def upload_video(
    video: UploadFile = File(...),
    email: str = Form(...),
    password: str = Form(...),
    session: AsyncSession = Depends(get_async_session)
):
    # 원본 파일 저장
    original_filename = video.filename
    original_filepath = os.path.join(UPLOAD_DIRECTORY, original_filename)
    
    with open(original_filepath, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # AI 분석 실행
    ai_result_dict = process_video_for_privacy(original_filepath, original_filename)

    # 에러 체크
    if "error" in ai_result_dict:
        raise HTTPException(status_code=500, detail=ai_result_dict["error"])

    # DB에 요청 정보 저장
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

    # 결과 응답
    return {
        "message": "분석 완료",
        "request_id": new_request.id,
        "filename": new_request.original_video_filename,
        "analysis": ai_result_dict["detection_summary"],
        "analyzed_video_url": ai_result_dict["analyzed_video_url"]
    }

#===============================================================================================================

# ID와 비밀번호로 분석 결과 가져오기 API
@app.post("/get-analysis/")
async def get_analysis(
    login_data: AnalysisLogin,
    session: AsyncSession = Depends(get_async_session)
):
    # DB에서 ID로 게시글 찾기
    statement = select(models.AnalysisRequest).where(
        models.AnalysisRequest.id == login_data.request_id
    )
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()

    # 데이터 존재 유무 및 비밀번호 확인
    if not db_post:
        raise HTTPException(status_code=404, detail="해당 ID의 분석 요청을 찾을 수 없습니다.")

    if not security.verify_password(login_data.password, db_post.password_hash):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")

    # 성공: 게시글 전체 정보를 React에게 반환 (DB에 저장된 그대로)
    return db_post