# import asyncio # 비동기 처리를 위함 (동시 작업 처리를 위해)
from fastapi import FastAPI, File, UploadFile, Depends, Form, HTTPException
# API : 서비스의 요청과 응답을 처리하는 기능
# FastAPI : 파이썬 웹 프레임 워크
# File, UploadFile : 파일 업로드를 처리하는 도구
# Depends : 의존성 주입 (DB 세션 등을 함수에 전달할 때 사용)
# Form : HTML 폼 데이터(이메일, 비밀번호 등)를 받기 위함
# HTTPException : 에러 발생 시 사용자에게 적절한 에러 코드를 보내기 위함

from fastapi.staticfiles import StaticFiles # 정적 파일(이미지, 영상 등)을 웹에서 접근 가능하게 하는 도구
from fastapi.middleware.cors import CORSMiddleware # 다른 도메인(React 등)에서 서버로 요청을 보낼 수 있게 허용하는 보안 설정 도구
from sqlmodel import SQLModel, select # DB 모델 정의 및 쿼리 작성을 위한 도구
from sqlalchemy.ext.asyncio import AsyncSession # 비동기 DB 처리를 위한 세션 도구
from pydantic import BaseModel # 데이터 검증을 위한 모델 도구 (입력 데이터 형식 정의)
from database import async_engine, create_db_and_tables, get_async_session # database.py에서 정의한 DB 연결 도구들 가져오기
import models # models.py에서 정의한 DB 테이블 구조 가져오기
import security # security.py에서 정의한 비밀번호 암호화 도구 가져오기
import shutil # 파일 저장(복사, 이동)을 위한 파일 관리 도구 라이브러리
import os # 파일 경로, 폴더 생성 등 운영체제 기능 사용 도구
import numpy as np # 행렬 연산 도구 (이미지 데이터 처리)
import requests # 인터넷에서 파일(AI 모델 등)을 다운로드하기 위한 도구
import bz2 # 압축 해제를 위한 라이브러리
import traceback # 에러 발생 시 자세한 원인을 출력하기 위한 도구

from ultralytics import YOLO # yolov8 라이브러리

import cv2  # OpenCV: 영상 처리 핵심 라이브러리
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

# AI 모델을 저장할 전역 변수
MODELS = {}

#===============================================================================================================

# CORS(교차 출처 리소스 공유) 설정
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
    # 서버 시작 시 AI 모델을 미리 로드
    print("AI 모델을 로딩합니다...")
    try:
        face_model_path, plate_model_path = check_and_download_files()
        MODELS['face'] = YOLO(face_model_path)
        MODELS['plate'] = YOLO(plate_model_path)
        MODELS['car'] = YOLO("yolov8m.pt")
        print("✅ AI 모델 로딩 완료.")
    except Exception as e:
        print(f"❌ AI 모델 로딩 실패: {e}")

#===============================================================================================================

# /get-analysis/ API가 받을 JSON 요청 본문 모델
# 결과 조회 시 클라이언트가 보내야 할 데이터 형식(ID는 숫자, 비번은 문자열)을 정의
class AnalysisLogin(BaseModel):
    request_id: int
    password: str

#===============================================================================================================

# AI 모델 파일 자동 다운로드 함수 (YOLOv8 Face 모델 및 번호판(license-plate) 모델)
def check_and_download_files():
    base_path = os.getcwd() # os.getcwd() : 현재 작업 경로를 반환
    
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
            except Exception as e: print(f"파일 삭제 실패: {e}")

    # DLL 청소
    if os.path.exists(dll_path):
        if os.path.getsize(dll_path) < 500000: # 500KB 미만이면 삭제
            try: os.remove(dll_path)
            except Exception as e: print(f"DLL 파일 삭제 실패: {e}")
    
    # 잘못된 버전의 DLL 파일 정리 (충돌 방지)
    for f in os.listdir(base_path):
        if f.startswith("openh264") and f.endswith(".dll") and f != target_dll:
            try: os.remove(os.path.join(base_path, f))
            except Exception as e: print(f"오래된 DLL 파일 삭제 실패: {e}")

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
        except Exception as e: print(f"얼굴 인식 모델 다운로드 실패: {e}")

#===============================================================================================================

    # 번호판(license-plate) 모델 다운로드
    if not os.path.exists(plate_model_path):
        print(f"번호판 모델 다운로드 중... ({plate_model_name})")
        try:
            # 1순위 오픈소스 번호판 모델 링크 (GitHub Raw)
            url = "https://raw.githubusercontent.com/ablanco1950/LicensePlate_Yolov8_MaxFilters/main/best.pt"
            r = requests.get(url, stream=True)

            if r.status_code == 200:
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

# 얼굴 = 타원, 번호판 = 직사각형으로 블러 처리 및 AI 분석
def process_video_for_privacy(video_path: str, original_filename: str) -> dict:

    try:
        print(f"'{video_path}' YOLOv8 얼굴 + 번호판 분석 시작.....")

        # 서버 시작 시 로드된 모델 가져오기
        face_model = MODELS.get('face')
        plate_model = MODELS.get('plate')
        car_model = MODELS.get('car')
        if not all([face_model, plate_model, car_model]):
            return {"error": "AI 모델이 로드되지 않았습니다. 서버 로그를 확인하세요."}

        # 영상 파일 열기
        cap = cv2.VideoCapture(video_path)  # OpenCV로 영상 파일 열기
        if not cap.isOpened():
            return {"error": "비디오 파일 열기 실패"}
        
        # 영상 정보 가져오기 (영상의 너비, 높이, 초당 프레임 수) # 나중에 다시 저장할 때 필요
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0.0: fps = 30.0 # 프레임 정보가 없으면 30으로 가정

        # 결과 파일명 및 경로 설정
        blurred_filename = f"blur_{original_filename}" # 결과 영상은 'blur_' 접두어 붙임
        blurred_filepath = os.path.join(UPLOAD_DIRECTORY, blurred_filename)
        
        # 영상 저장 설정 (코덱)
        # avc1 (웹 호환성이 좋음)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # 비디오 압축 방식
        out = cv2.VideoWriter(blurred_filepath, fourcc, fps, (frame_width, frame_height))

        # 얼굴과 번호판 카운트를 분리
        faces_blurred_count = 0
        plates_blurred_count = 0

        frame_count = 0 # 현재 처리 중인 프레임 번호

        # 번호판 기억 저장소 (개선된 구조)
        # 구조: {track_id: {'coords': (x1,y1,x2,y2), 'velocity': (vx,vy), 'life': int, 'confidence_history': [], 'last_seen': frame_num}}
        plate_memory = {}
        
        # 칼만 필터 기반 위치 예측을 위한 속도 저장
        # 이전 프레임의 차량 위치 저장 (움직임 예측용)
        prev_car_positions = {}

        # 번호판 검증 함수 (강화된 버전)
        def is_valid_plate(x1, y1, x2, y2, frame_w, frame_h):
            w = x2 - x1
            h = y2 - y1

            # 높이가 0이거나 이상하면 바로 무시
            if h <= 0 or w <= 0: 
                return False

            # 노이즈 제거 (너무 작은 건 무시)
            if w < 15 or h < 8: return False

            # 보닛 필터
            if y2 > frame_h * 0.95 and w > frame_w * 0.30:
                return False

            # 비율 검사 (한국 번호판: 가로가 세로의 약 2~2.5배)
            aspect_ratio = w / h
            
            # 번호판 비율: 1.5 ~ 5.0 (너무 길거나 정사각형에 가까우면 제외)
            if aspect_ratio < 1.5 or aspect_ratio > 5.0:
                return False
        
            # 크기 검사 (화면의 2% 이상이면 오탐지 가능성 높음)
            plate_area = w * h
            frame_area = frame_w * frame_h
            if plate_area / frame_area > 0.02:
                return False
            
            # 너무 세로로 긴 건 측면 광고일 가능성
            if h > w * 0.8:  # 세로가 가로의 80% 이상이면 제외
                return False

            return True
        
        # 번호판 좌표 스무딩 함수 (떨림 방지)
        def smooth_coordinates(new_coords, old_coords, alpha=0.3):
            """새 좌표와 이전 좌표를 보간하여 부드럽게 만듦"""
            if old_coords is None:
                return new_coords
            nx1, ny1, nx2, ny2 = new_coords
            ox1, oy1, ox2, oy2 = old_coords
            return (
                int(alpha * nx1 + (1 - alpha) * ox1),
                int(alpha * ny1 + (1 - alpha) * oy1),
                int(alpha * nx2 + (1 - alpha) * ox2),
                int(alpha * ny2 + (1 - alpha) * oy2)
            )
        
        # 움직임 예측 함수 (속도 기반)
        def predict_position(coords, velocity, frames_ahead=1):
            """속도를 기반으로 미래 위치 예측"""
            if velocity is None:
                return coords
            x1, y1, x2, y2 = coords
            vx, vy = velocity
            
            # 속도 상한선 (비정상적으로 큰 속도 방지)
            max_speed = 50  # 픽셀/프레임
            vx = max(-max_speed, min(max_speed, vx))
            vy = max(-max_speed, min(max_speed, vy))
            
            return (
                int(x1 + vx * frames_ahead),
                int(y1 + vy * frames_ahead),
                int(x2 + vx * frames_ahead),
                int(y2 + vy * frames_ahead)
            )
        
        # 확장된 블러 영역 계산 함수 (움직임 고려)
        def get_expanded_blur_region(coords, velocity, frame_w, frame_h, expansion_ratio=0.25):
            """속도를 고려하여 블러 영역을 확장 (움직이는 물체의 잔상 커버)"""
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            
            # 원본 크기가 비정상적이면 무시 (화면의 5% 이상이면 오류)
            if w * h > frame_w * frame_h * 0.05:
                return (x1, y1, x2, y2)  # 확장 없이 반환
            
            # 기본 확장 (최대 30픽셀로 제한)
            pad_w = min(int(w * expansion_ratio), 30)
            pad_h = min(int(h * expansion_ratio), 30)
            
            # 속도가 있으면 움직이는 방향으로 추가 확장 (최대 20픽셀)
            if velocity:
                vx, vy = velocity
                speed = (vx**2 + vy**2) ** 0.5
                if speed > 5:
                    extra_pad = min(int(speed * 0.3), 20)  # 최대 20픽셀
                    pad_w += extra_pad
                    pad_h += extra_pad
            
            # 확장된 좌표 (화면 범위 내로 제한)
            ex1 = max(0, x1 - pad_w)
            ey1 = max(0, y1 - pad_h)
            ex2 = min(frame_w, x2 + pad_w)
            ey2 = min(frame_h, y2 + pad_h)
            
            # 최종 크기가 원본의 3배를 넘으면 원본 크기로 제한
            final_w, final_h = ex2 - ex1, ey2 - ey1
            if final_w > w * 3 or final_h > h * 3:
                return (x1, y1, x2, y2)  # 확장 없이 반환
            
            return (ex1, ey1, ex2, ey2)
        
        # 프레임 스킵 설정 (얼굴만 스킵, 번호판은 매 프레임 탐지)
        FACE_SKIP_FRAMES = 2  # 얼굴: 3프레임당 1번 탐지
        
        # 얼굴 메모리 (프레임 스킵 시 블러 유지용)
        face_memory = {}
        
        # 영상이 끝날 때까지 프레임 반복 처리 시작
        while cap.isOpened():
            success, frame = cap.read() # 한 프레임 읽기
            if not success: break # 영상 끝나면 종료

            # 진행 상황 로그 출력 (30프레임마다)
            frame_count += 1 
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...", end='\r')

            # 얼굴 탐지 여부 (프레임 스킵 - 얼굴만)
            do_face_detection = (frame_count % (FACE_SKIP_FRAMES + 1) == 1)
            
            # === 차량 탐지 (매 프레임) - 해상도 높여서 버스도 잡기 ===
            car_results = car_model(frame, classes=[2, 3, 5, 7], imgsz=960, verbose=False, conf=0.03)
            
            car_boxes = []
            if car_results:
                for r in car_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        car_boxes.append((coords, cls_id))
            
            # === 얼굴 탐지 (스킵 적용) ===
            if do_face_detection:
                face_results = face_model.track(frame, conf=0.20, imgsz=1280, augment=False, persist=True, tracker="botsort.yaml", verbose=False)

                if face_results:
                    for result in face_results:
                        if result is None or not hasattr(result, 'boxes') or result.boxes is None: continue

                        for box in result.boxes:
                            face_track_id = int(box.id.item() if box.id is not None else -1)
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            
                            face_w = x2 - x1
                            face_h = y2 - y1
                            if (face_w * face_h) > (frame_width * frame_height * 0.025):
                                continue

                            face_aspect_ratio = face_w / face_h if face_h > 0 else 0
                            if face_aspect_ratio > 1.2 or face_aspect_ratio < 0.25:
                                continue

                            w, h = x2 - x1, y2 - y1
                            pad_x = int(w * 0.1)
                            pad_y_top = int(h * 0.2)
                            pad_y_bot = int(h * 0.2)
                            
                            bx1 = max(0, x1 - pad_x)
                            by1 = max(0, y1 - pad_y_top)
                            bx2 = min(frame_width, x2 + pad_x)
                            by2 = min(frame_height, y2 + pad_y_bot)
                            
                            if face_track_id != -1:
                                face_memory[face_track_id] = {
                                    'coords': (bx1, by1, bx2, by2),
                                    'life': 10
                                }
                            
                            roi = frame[by1:by2, bx1:bx2]
                            if roi.size == 0: continue
                            
                            try:
                                kw = int((bx2-bx1)/1.5) | 1
                                kh = int((by2-by1)/1.5) | 1
                                blurred = cv2.GaussianBlur(roi, (kw, kh), 0)
                                
                                mask = np.zeros_like(roi)
                                cv2.ellipse(mask, ((bx2-bx1)//2, (by2-by1)//2), ((bx2-bx1)//2, (by2-by1)//2), 0, 0, 360, (255, 255, 255), -1)
                                frame[by1:by2, bx1:bx2] = np.where(mask > 0, blurred, roi)
                                faces_blurred_count += 1
                            except:
                                try:
                                    kw = int((bx2-bx1)/1.5) | 1
                                    kh = int((by2-by1)/1.5) | 1
                                    frame[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (kw, kh), 0)
                                    faces_blurred_count += 1
                                except: pass
            else:
                # 얼굴 스킵 프레임: 메모리 기반 블러
                face_keys_to_remove = []
                for fid, finfo in face_memory.items():
                    fx1, fy1, fx2, fy2 = finfo['coords']
                    roi = frame[fy1:fy2, fx1:fx2]
                    if roi.size > 0:
                        try:
                            kw = int((fx2-fx1)/1.5) | 1
                            kh = int((fy2-fy1)/1.5) | 1
                            blurred = cv2.GaussianBlur(roi, (kw, kh), 0)
                            mask = np.zeros_like(roi)
                            cv2.ellipse(mask, ((fx2-fx1)//2, (fy2-fy1)//2), ((fx2-fx1)//2, (fy2-fy1)//2), 0, 0, 360, (255, 255, 255), -1)
                            frame[fy1:fy2, fx1:fx2] = np.where(mask > 0, blurred, roi)
                        except: pass
                    finfo['life'] -= 1
                    if finfo['life'] <= 0:
                        face_keys_to_remove.append(fid)
                for fk in face_keys_to_remove:
                    del face_memory[fk]

            # === 번호판 탐지 (매 프레임 - 스킵 없음) ===
            plate_results = plate_model.track(frame, conf=0.08, imgsz=1920, augment=False, persist=True, tracker="botsort.yaml", verbose=False)

            current_frame_ids = [] # 이번 프레임에서 잡은 번호판 ID들
            detected_plates_this_frame = {}  # {track_id: coords} 이번 프레임 탐지 결과

            if plate_results:
                for result in plate_results:
                    # 결과 유효성 검사
                    if result is None or not hasattr(result, 'boxes') or result.boxes is None: continue

                    for box in result.boxes:
                        
                        # 트래킹 ID 가져오기
                        track_id = int(box.id.item() if box.id is not None else -1)

                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # 번호판 검증 통과한 번호판만 처리
                        if not is_valid_plate(x1, y1, x2, y2, frame_width, frame_height):
                            continue

                        # 번호판이 자동차 박스 안에 있는지 검사
                        valid_loc = False
                        p_cx = (x1 + x2) / 2
                        p_cy = (y1 + y2) / 2
                        plate_w = x2 - x1

                        if len(car_boxes) > 0:
                            for c_data in car_boxes:
                                c_box, c_cls = c_data
                                cx1, cy1, cx2, cy2 = c_box
                                c_w = cx2 - cx1; c_h = cy2 - cy1
                                c_cx = (cx1 + cx2) / 2
                                
                                # 번호판 중심이 차량 영역 안에 있는지 (패딩 20%)
                                pad_w = c_w * 0.20; pad_h = c_h * 0.20
                                if not ((cx1 - pad_w) < p_cx < (cx2 + pad_w) and 
                                        (cy1 - pad_h) < p_cy < (cy2 + pad_h)):
                                    continue
                                
                                # === 측면 광고 필터 (핵심!) ===
                                # 번호판이 차량 중앙에서 너무 벗어나면 측면 광고
                                if abs(p_cx - c_cx) > (c_w * 0.35):
                                    continue
                                
                                # 번호판이 너무 크면 오탐지 (차량 너비의 45% 이하만 허용)
                                if plate_w > (c_w * 0.45):
                                    continue
                                
                                # === 상단 필터 (완화) ===
                                # 버스/트럭: 상단 25%만 무시 (앞번호판 허용)
                                if c_cls in [5, 7]:
                                    if p_cy < (cy1 + c_h * 0.25): continue
                                else:
                                    # 승용차: 상단 15%만 무시
                                    if p_cy < (cy1 + c_h * 0.15): continue
                                
                                valid_loc = True
                                break
                        
                        # 차량 없어도 메모리에 있으면 계속 추적
                        if not valid_loc and track_id != -1 and track_id in plate_memory:
                            valid_loc = True
                        
                        # 차량 미탐지 + 메모리에도 없지만, 번호판이 명확한 경우 (추가 검증)
                        if not valid_loc and len(car_boxes) == 0:
                            # 번호판 특성이 매우 명확한 경우에만 허용
                            plate_h = y2 - y1
                            plate_aspect = plate_w / plate_h if plate_h > 0 else 0
                            # 한국 번호판 전형적 비율: 2.0~3.5 (매우 엄격)
                            # 크기도 적당해야 함 (너무 크거나 작으면 제외)
                            plate_area = plate_w * plate_h
                            frame_area = frame_width * frame_height
                            area_ratio = plate_area / frame_area
                            
                            if (2.0 <= plate_aspect <= 3.5 and 
                                0.0005 < area_ratio < 0.008 and
                                p_cy > frame_height * 0.3):  # 화면 상단 30%는 제외
                                valid_loc = True
                        
                        if not valid_loc:
                            continue

                        # 현재 좌표
                        current_coords = (x1, y1, x2, y2)
                        
                        # 트래킹 ID가 있고 메모리에 있으면 스무딩 및 속도 계산
                        velocity = (0, 0)
                        if track_id != -1 and track_id in plate_memory:
                            old_info = plate_memory[track_id]
                            old_coords = old_info.get('coords', current_coords)
                            
                            # 속도 계산 (이전 위치와 현재 위치 차이)
                            velocity = (
                                (x1 + x2) / 2 - (old_coords[0] + old_coords[2]) / 2,
                                (y1 + y2) / 2 - (old_coords[1] + old_coords[3]) / 2
                            )
                            
                            # 좌표 스무딩 (떨림 방지) - alpha가 낮을수록 부드럽게
                            smoothed_coords = smooth_coordinates(current_coords, old_coords, alpha=0.4)
                            x1, y1, x2, y2 = smoothed_coords

                        plates_blurred_count += 1

                        # 번호판 영역 확장 (움직임 고려하여 더 넓게)
                        blur_region = get_expanded_blur_region(
                            (x1, y1, x2, y2), 
                            velocity,
                            frame_width, frame_height,
                            expansion_ratio=0.20  # 20% 확장
                        )
                        px1, py1, px2, py2 = blur_region

                        # 1. 블러 처리
                        roi = frame[py1:py2, px1:px2]
                        if roi.size == 0: continue

                        try:
                            # 블러 정도 (더 강하게)
                            kw = int((px2-px1)/1.5) | 1
                            kh = int((py2-py1)/1.5) | 1
                            frame[py1:py2, px1:px2] = cv2.GaussianBlur(roi, (kw, kh), 0)
                        except Exception as e: print(f"번호판 블러 처리 중 오류: {e}")

                        # 2. 메모리에 저장 (속도 정보 포함)
                        if track_id != -1:
                            plate_memory[track_id] = {
                                'coords': (px1, py1, px2, py2),
                                'velocity': velocity,
                                'life': 90,  # 3초(30fps 기준) 동안 기억 유지
                                'last_seen': frame_count,
                                'confidence_streak': plate_memory.get(track_id, {}).get('confidence_streak', 0) + 1
                            }
                            current_frame_ids.append(track_id)
                            detected_plates_this_frame[track_id] = (px1, py1, px2, py2)

            # 놓친 번호판 블러 처리 (움직임 예측 기반)
            keys_to_remove = []
            for tid, info in plate_memory.items():
                if tid not in current_frame_ids: # 방금 놓쳤다면
                    # 속도 기반으로 위치 예측
                    old_coords = info['coords']
                    velocity = info.get('velocity', (0, 0))
                    frames_since_seen = frame_count - info.get('last_seen', frame_count)
                    
                    # 너무 오래 전에 본 경우 삭제 (예측이 부정확해짐)
                    if frames_since_seen > 15:
                        keys_to_remove.append(tid)
                        continue
                    
                    # 예측 위치 계산 (속도 * 놓친 프레임 수, 최대 3프레임까지만)
                    predicted_coords = predict_position(old_coords, velocity, min(frames_since_seen, 3))
                    lx1, ly1, lx2, ly2 = predicted_coords

                    # 화면 밖 체크
                    lx1, ly1 = max(0, lx1), max(0, ly1)
                    lx2, ly2 = min(frame_width, lx2), min(frame_height, ly2)
                    
                    # 예측 위치가 화면 밖으로 나가면 삭제
                    if lx2 <= lx1 or ly2 <= ly1:
                        keys_to_remove.append(tid)
                        continue
                    
                    # 블러 영역 크기 검증 (화면의 3% 초과하면 무시)
                    blur_area = (lx2 - lx1) * (ly2 - ly1)
                    if blur_area > frame_width * frame_height * 0.03:
                        keys_to_remove.append(tid)
                        continue

                    # 블러 영역 확장 (움직임 기반, 제한적으로)
                    expanded_region = get_expanded_blur_region(
                        (lx1, ly1, lx2, ly2), velocity, frame_width, frame_height, 0.15
                    )
                    lx1, ly1, lx2, ly2 = expanded_region
                    
                    # 확장 후에도 크기 재검증
                    blur_area = (lx2 - lx1) * (ly2 - ly1)
                    if blur_area > frame_width * frame_height * 0.03:
                        keys_to_remove.append(tid)
                        continue

                    roi = frame[ly1:ly2, lx1:lx2]
                    if roi.size > 0:
                        try:
                            # 블러 정도
                            kw = int((lx2-lx1)/1.5) | 1
                            kh = int((ly2-ly1)/1.5) | 1
                            blurred_plate = cv2.GaussianBlur(roi, (kw, kh), 0)
                            frame[ly1:ly2, lx1:lx2] = blurred_plate
                        except: pass # 메모리 블러는 실패해도 조용히 넘어감
                    
                    # 예측 좌표를 메모리에 업데이트 (원본 좌표만, 확장된 건 저장 안함)
                    original_w = old_coords[2] - old_coords[0]
                    original_h = old_coords[3] - old_coords[1]
                    info['coords'] = predicted_coords  # 확장 전 좌표 저장
                    
                    # 수명 감소 (0이 되면 삭제)
                    info['life'] -= 1
                    if info['life'] <= 0:
                        keys_to_remove.append(tid)

            # 수명 다한 기억 삭제
            for k in keys_to_remove:
                del plate_memory[k]

            # 멀리 있는 차량도 탐지
            # 차를 잘라내서(Crop) 확대해서 봄
            for c_data in car_boxes:
                cbox_coords, c_cls = c_data
                cx1, cy1, cx2, cy2 = [int(c) for c in cbox_coords]

                cw, ch = cx2 - cx1, cy2 - cy1

                # 차량 영역 자르기
                pad_w = int(cw * 0.20)
                pad_h = int(ch * 0.20)
                bx1 = max(0, cx1 - pad_w)
                by1 = max(0, cy1 - pad_h)
                bx2 = min(frame_width, cx2 + pad_w)
                by2 = min(frame_height, cy2 + pad_h)

                car_crop = frame[by1:by2, bx1:bx2]
                if car_crop.size == 0: continue

                # 차가 작으면 이미지를 3배 확대해서 보여줌
                try:
                    input_crop = car_crop
                    if car_crop.shape[1] < 200: 
                        input_crop = cv2.resize(car_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                except: input_crop = car_crop

                # 자른 이미지에서 번호판 찾기 (확대한 상태에서 찾음)
                zoom_results = plate_model.predict(input_crop, conf=0.1, imgsz=640, verbose=False)

                if zoom_results:
                    for z_result in zoom_results:
                        if z_result.boxes is None: continue
                        for z_box in z_result.boxes:
                            zx1, zy1, zx2, zy2 = map(int, z_box.xyxy[0].cpu().numpy())

                            # 확대한 경우 비율에 맞춰 다시 줄여야 함
                            if car_crop.shape[1] < 200:
                                zx1, zy1, zx2, zy2 = int(zx1/3), int(zy1/3), int(zx2/3), int(zy2/3)

                            # 원래 프레임 좌표로 변환
                            gx1 = bx1 + zx1
                            gy1 = by1 + zy1
                            gx2 = bx1 + zx2
                            gy2 = by1 + zy2
                                
                            # 좌표가 화면 밖으로 나가지 않게 조절
                            gx1 = max(0, gx1); gy1 = max(0, gy1)
                            gx2 = min(frame_width, gx2); gy2 = min(frame_height, gy2)

                            # 모양 검사(번호판 검증 통과한 것만 블러 처리)
                            if not is_valid_plate(gx1, gy1, gx2, gy2, frame_width, frame_height):
                                continue
                            
                            # 위치 검사 
                            p_cy = (gy1 + gy2) / 2
                            p_cx = (gx1 + gx2) / 2
                            c_cx = (cx1 + cx2) / 2

                            # 차종별 필터
                            # 버스/트럭은 상단 70% 무시
                            limit_ratio = 0.70 if c_cls in [5, 7] else 0.30
                            if (p_cy - cy1) < (ch * limit_ratio): continue 
                            
                            # 너비 필터
                            # 확대해서 찾았더라도, 원래 차 크기 대비 50% 이상이면 가짜 (거대 블러 방지)
                            if (gx2 - gx1) > (cw * 0.50): continue

                            roi = frame[gy1:gy2, gx1:gx2]
                            if roi.size > 0:
                                try:
                                    kw = int((gx2-gx1)/2) | 1; kh = int((gy2-gy1)/2) | 1
                                    frame[gy1:gy2, gx1:gx2] = cv2.GaussianBlur(roi, (kw, kh), 0)
                                    plates_blurred_count += 1
                                except Exception as e: print(f"확대 블러 처리 중 오류: {e}")

            # 처리된 프레임을 비디오 파일에 저장(녹화)
            out.write(frame)

        total_blurred_objects = faces_blurred_count + plates_blurred_count
        cap.release() # 영상 파일 닫기
        out.release() # 저장 완료

        print(f"'{blurred_filepath}' YOLO 분석 완료. (총 블러 처리 객체: {total_blurred_objects})")
        
        # 결과 반환
        return {
            "detection_summary": {
                "faces_blurred": faces_blurred_count,
                "plates_blurred": plates_blurred_count,
                "total_blurred_count": total_blurred_objects
            },
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