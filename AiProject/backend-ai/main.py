# (신규) .env 파일을 읽기 위해 load_dotenv를 임포트
from dotenv import load_dotenv
import os  # 파일 경로, 폴더 생성 등 운영체제 기능 사용 도구
import sys
import json  # JSON 처리를 위해 추가

# 🚨 [중요] OpenCV가 DLL을 찾을 수 있도록 경로 설정을 가장 먼저 실행해야 합니다.
load_dotenv()

# 현재 작업 경로를 시스템 경로에 추가 (openh264-1.8.0-win64.dll 찾기 위함)
if hasattr(os, 'add_dll_directory'):
    try:
        os.add_dll_directory(os.getcwd())
    except Exception:
        pass
os.environ['PATH'] = os.getcwd() + ';' + os.environ['PATH']

# 라이브러리 임포트
from fastapi import FastAPI, File, UploadFile, Depends, Form, HTTPException, Path, Body
# FastAPI : 파이썬 웹 프레임 워크
# File, UploadFile : 파일 업로드를 처리하는 도구
# Depends : 의존성 주입 (DB 세션 등을 함수에 전달할 때 사용)
# Form : HTML 폼 데이터(이메일, 비밀번호 등)를 받기 위함
# HTTPException : 에러 발생 시 사용자에게 적절한 에러 코드를 보내기 위함

from fastapi.middleware.cors import CORSMiddleware  # 다른 도메인(React 등)에서 서버로 요청을 보낼 수 있게 허용하는 보안 설정 도구
from fastapi.staticfiles import StaticFiles  # 정적 파일(이미지, 영상 등)을 웹에서 접근 가능하게 하는 도구
from fastapi.responses import FileResponse  # 파일 다운로드 응답을 위한 도구
from sqlmodel import SQLModel, Session, select, func  # DB 모델 정의 및 쿼리 작성을 위한 도구
from sqlalchemy import or_ 
from sqlalchemy.ext.asyncio import AsyncSession  # 비동기 DB 처리를 위한 세션 도구
from pydantic import BaseModel  # 데이터 검증을 위한 모델 도구 (입력 데이터 형식 정의)
from typing import List, Optional
from datetime import datetime
import uuid 
import asyncio 
import time  # CPU 양보
from database import async_engine, create_db_and_tables, get_async_session, async_session  # database.py에서 정의한 DB 연결 도구들 가져오기
import models  # models.py에서 정의한 DB 테이블 구조 가져오기
import security  # security.py에서 정의한 비밀번호 암호화 도구 가져오기
import shutil  # 파일 저장(복사, 이동)을 위한 파일 관리 도구 라이브러리
import cv2  # OpenCV: 영상 처리 핵심 라이브러리
import numpy as np  # 행렬 연산 도구 (이미지 데이터 처리)
import requests  # 인터넷에서 파일(AI 모델 등)을 다운로드하기 위한 도구
import bz2  # 압축 해제를 위한 라이브러리
import traceback  # 에러 발생 시 자세한 원인을 출력하기 위한 도구
from ultralytics import YOLO  # YOLOv8 라이브러리 (객체 탐지 AI 모델) 

# --- 🚨 상태 상수 정의 ---
STATUS_PENDING = "PENDING"
STATUS_IN_PROGRESS = "IN_PROGRESS" 
STATUS_COMPLETED = "COMPLETED"
ADMIN_PASSWORD = "1234" 

# --- 전역 변수: 분석 중지 시그널 관리 ---
# {post_id: True} 형태로 저장되면 해당 ID의 분석을 중단함
STOP_SIGNALS = {}

# --- 전역 변수: AI 모델 (서버 시작 시 1회 로드) ---
# 매번 로드하면 느리므로, 서버 시작 시 메모리에 상주시킴
MODELS = {}

# --- Pydantic 모델 ---
class PostResponse(BaseModel):
    id: int
    title: str
    author: str
    status: str
    created_at: datetime
    target_address: str 
    
    class Config:
        from_attributes = True 

class PostDetailResponse(PostResponse):
    content: Optional[str]
    email: str
    analyzed_video_path: Optional[str] 
    original_video_filename: Optional[str]
    
    class Config:
        from_attributes = True

class PaginatedPostResponse(BaseModel):
    total_posts: int
    total_pages: int
    posts: List[PostResponse]

class PasswordCheck(BaseModel):
    password: str

# 글 수정용 모델
class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    target_address: Optional[str] = None
    password: str # 본인 확인용

# --- uploads 폴더 설정 ---
UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIRECTORY), name="uploads")

# ---------------------------------------------------------
# [서버 재시작 시 좀비 상태 복구 로직]
# ---------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    await create_db_and_tables()
    
    print("🔄 서버 시작: 비정상 종료된 작업이 있는지 확인합니다...")
    async with async_session() as session:
        try:
            statement = select(models.AnalysisRequest).where(models.AnalysisRequest.status == STATUS_IN_PROGRESS)
            result = await session.execute(statement)
            stuck_posts = result.scalars().all()

            if stuck_posts:
                count = len(stuck_posts)
                print(f"⚠️ 비정상 종료된 작업 {count}건 발견! '대기 중' 상태로 복구합니다.")
                for post in stuck_posts:
                    post.status = STATUS_PENDING
                    session.add(post)
                await session.commit()
                print("✅ 복구 완료.")
            else:
                print("✅ 비정상 종료된 작업 없음.")
        except Exception as e:
            print(f"❌ 초기화 중 오류 발생: {e}")
    
    # --- [추가] 서버 시작 시 AI 모델 미리 로드 (성능 향상) ---
    print("🤖 AI 모델을 로딩합니다...")
    try:
        face_model_path, plate_model_path = check_and_download_files()
        MODELS['face'] = YOLO(face_model_path)   # 얼굴 인식 모델
        MODELS['plate'] = YOLO(plate_model_path) # 번호판 인식 모델
        MODELS['car'] = YOLO("yolov8m.pt")       # 차량 인식 모델 (자동 다운로드)
        print("✅ AI 모델 로딩 완료.")
    except Exception as e:
        print(f"❌ AI 모델 로딩 실패: {e}")

@app.get("/")
def read_root():
    return {"Hello": "Backend"}

# -----------------------------------------------
# --- [업데이트] AI 모델 관리 및 분석 로직 ---
# -----------------------------------------------

def check_and_download_files():
    """기본 모델 파일 및 코덱 DLL 확인 및 다운로드"""
    base_path = os.getcwd()
    
    face_model_name = "yolov8n-face.pt"
    face_model_path = os.path.join(base_path, face_model_name)

    plate_model_name = "yolov8n-license-plate.pt"
    plate_model_path = os.path.join(base_path, plate_model_name)

    target_dll = "openh264-1.8.0-win64.dll"
    dll_path = os.path.join(base_path, target_dll)
    
    # 1. 얼굴 모델
    if not os.path.exists(face_model_path):
        try:
            url = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"
            r = requests.get(url, stream=True)
            with open(face_model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except: pass

    # 2. 번호판 모델
    if not os.path.exists(plate_model_path):
        try:
            # 1순위
            url = "https://raw.githubusercontent.com/ablanco1950/LicensePlate_Yolov8_MaxFilters/main/best.pt"
            r = requests.get(url, stream=True)
            if r.status_code == 200:
                with open(plate_model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            else:
                # 2순위 백업
                url2 = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"
                r2 = requests.get(url2, stream=True)
                with open(plate_model_path, 'wb') as f:
                    for chunk in r2.iter_content(chunk_size=8192): f.write(chunk)
        except: pass
            
    # 3. DLL
    if not os.path.exists(dll_path):
        try:
            url = "http://ciscobinary.openh264.org/openh264-1.8.0-win64.dll.bz2"
            r = requests.get(url, stream=True)
            decompressed_data = bz2.decompress(r.content)
            with open(dll_path, 'wb') as f: f.write(decompressed_data)
        except: pass

    # 차량 모델은 ultralytics가 자동 다운로드하므로 경로 리턴만 함 (혹은 이름만 리턴)
    return face_model_path, plate_model_path

# 🚨 분석 함수 업데이트 (팀원 코드 통합)
def process_video_for_privacy(video_path: str, post_id: int) -> dict:
    try:
        print(f"AI 정밀 분석 시작 (Post ID: {post_id}): {video_path}")
        
        # 서버 시작 시 로드된 전역 모델 가져오기 (성능 향상)
        face_model = MODELS.get('face')
        plate_model = MODELS.get('plate')
        car_model = MODELS.get('car')
        
        # 모델이 로드되지 않았으면 에러 반환
        if not all([face_model, plate_model, car_model]):
            return {"error": "AI 모델이 로드되지 않았습니다. 서버 로그를 확인하세요."}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": "비디오 파일을 열 수 없습니다."}

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps == 0.0: fps = 30.0

        filename_with_uuid = os.path.basename(video_path)
        parts = filename_with_uuid.split('_', 1)
        original_name_ext = parts[1] if len(parts) > 1 else filename_with_uuid
        name, ext = os.path.splitext(original_name_ext)
        now_str = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # 임시 파일은 AVI로 저장 (OpenCV 호환성), 최종 파일은 MP4
        temp_filename = f"{name}_blurred_{now_str}_temp.avi"
        temp_filepath = os.path.join(UPLOAD_DIRECTORY, temp_filename)
        blurred_filename = f"{name}_blurred_{now_str}.mp4"
        blurred_filepath = os.path.join(UPLOAD_DIRECTORY, blurred_filename)

        # OpenCV VideoWriter (XVID 코덱으로 AVI 저장)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_filepath, fourcc, fps, (frame_width, frame_height))

        # --- 헬퍼 함수들 (팀원 코드에서 가져옴) ---
        def is_valid_plate(x1, y1, x2, y2, frame_w, frame_h):
            w, h = x2 - x1, y2 - y1
            if h <= 0 or w <= 0: return False
            if w < 15 or h < 8: return False # 노이즈 제거
            if y2 > frame_h * 0.95 and w > frame_w * 0.30: return False # 보닛 필터
            aspect_ratio = w / h
            if aspect_ratio < 1.5 or aspect_ratio > 5.0: return False # 비율 검사
            if (w * h) / (frame_w * frame_h) > 0.02: return False # 너무 크면 제외
            if h > w * 0.8: return False # 세로로 긴 것 제외
            return True

        def smooth_coordinates(new_coords, old_coords, alpha=0.3):
            if old_coords is None: return new_coords
            nx1, ny1, nx2, ny2 = new_coords
            ox1, oy1, ox2, oy2 = old_coords
            return (
                int(alpha * nx1 + (1 - alpha) * ox1),
                int(alpha * ny1 + (1 - alpha) * oy1),
                int(alpha * nx2 + (1 - alpha) * ox2),
                int(alpha * ny2 + (1 - alpha) * oy2)
            )

        def predict_position(coords, velocity, frames_ahead=1):
            if velocity is None: return coords
            x1, y1, x2, y2 = coords
            vx, vy = velocity
            max_speed = 50
            vx = max(-max_speed, min(max_speed, vx))
            vy = max(-max_speed, min(max_speed, vy))
            return (
                int(x1 + vx * frames_ahead), int(y1 + vy * frames_ahead),
                int(x2 + vx * frames_ahead), int(y2 + vy * frames_ahead)
            )

        def get_expanded_blur_region(coords, velocity, frame_w, frame_h, expansion_ratio=0.25):
            x1, y1, x2, y2 = coords
            w, h = x2 - x1, y2 - y1
            if w * h > frame_w * frame_h * 0.05: return (x1, y1, x2, y2)
            pad_w = min(int(w * expansion_ratio), 30)
            pad_h = min(int(h * expansion_ratio), 30)
            if velocity:
                vx, vy = velocity
                speed = (vx**2 + vy**2) ** 0.5
                if speed > 5:
                    extra_pad = min(int(speed * 0.3), 20)
                    pad_w += extra_pad
                    pad_h += extra_pad
            ex1 = max(0, x1 - pad_w)
            ey1 = max(0, y1 - pad_h)
            ex2 = min(frame_w, x2 + pad_w)
            ey2 = min(frame_h, y2 + pad_h)
            return (ex1, ey1, ex2, ey2)

        face_count = 0
        plate_count = 0
        frame_idx = 0
        
        # 메모리 변수
        plate_memory = {}
        
        while cap.isOpened():
            # 🚨 중지 신호 확인
            if STOP_SIGNALS.get(post_id):
                print(f"🛑 분석 중지 요청 감지! (Post ID: {post_id})")
                cap.release(); out.release()
                if os.path.exists(blurred_filepath): os.remove(blurred_filepath)
                return {"stopped": True}

            time.sleep(0.001)
            success, frame = cap.read()
            if not success: break
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"[ID:{post_id}] Frame {frame_idx}/{total_frames} ({progress:.1f}%) - Obj: {face_count+plate_count}", end='\r', flush=True)

            # 1. 차량 탐지 (번호판 필터링용) - 해상도 높여서 버스도 잡기
            # classes: 2=car, 3=motorcycle, 5=bus, 7=truck
            # conf=0.03: 매우 민감하게 설정하여 멀리 있는 차량도 탐지
            car_results = car_model(frame, classes=[2, 3, 5, 7], imgsz=960, verbose=False, conf=0.03)
            car_boxes = []
            if car_results:
                for r in car_results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0].item())
                        car_boxes.append((coords, cls_id))

            # 2. 얼굴 인식 - 고해상도로 정밀 탐지 (imgsz=1280)
            face_results = face_model.track(frame, conf=0.20, imgsz=1280, augment=False, persist=True, tracker="botsort.yaml", verbose=False)
            if face_results:
                for result in face_results:
                    if result.boxes:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            face_w, face_h = x2 - x1, y2 - y1
                            
                            # 너무 크거나 비율 이상하면 패스
                            if (face_w * face_h) > (frame_width * frame_height * 0.05): continue
                            face_aspect = face_w / face_h if face_h > 0 else 0
                            if face_aspect > 1.2 or face_aspect < 0.25: continue

                            # 타원형 블러 적용 (안전한 커널 크기 적용)
                            try:
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                axes = (int(face_w * 0.5), int(face_h * 0.6))
                                mask = np.zeros_like(frame)
                                cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, (255, 255, 255), -1)
                                # 안전한 커널
                                kw = max(3, (int((x2 - x1) / 1.5) | 1))
                                kh = max(3, (int((y2 - y1) / 1.5) | 1))
                                blurred_part = cv2.GaussianBlur(frame, (kw, kh), 0)
                                frame = np.where(mask > 0, blurred_part, frame)
                                face_count += 1
                            except: pass

            # 3. 번호판 인식 (고급 로직) - 최고 해상도로 정밀 탐지 (imgsz=1920)
            plate_results = plate_model.track(frame, conf=0.08, imgsz=1920, augment=False, persist=True, tracker="botsort.yaml", verbose=False)
            current_frame_ids = []
            
            if plate_results:
                for result in plate_results:
                    if result.boxes:
                        for box in result.boxes:
                            track_id = int(box.id.item() if box.id is not None else -1)
                            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                            
                            # 기본 검증
                            if not is_valid_plate(x1, y1, x2, y2, frame_width, frame_height): continue

                            # 차량 내부 확인 및 필터링
                            valid_loc = False
                            p_cx, p_cy = (x1+x2)/2, (y1+y2)/2
                            p_w = x2 - x1
                            
                            if len(car_boxes) > 0:
                                for c_data in car_boxes:
                                    c_box, c_cls = c_data
                                    cx1, cy1, cx2, cy2 = c_box
                                    c_w, c_h = cx2 - cx1, cy2 - cy1
                                    c_cx = (cx1 + cx2) / 2
                                    
                                    # 차량 영역 패딩 (20%)
                                    pad_w, pad_h = c_w * 0.20, c_h * 0.20
                                    if not ((cx1 - pad_w) < p_cx < (cx2 + pad_w) and 
                                            (cy1 - pad_h) < p_cy < (cy2 + pad_h)):
                                        continue
                                    
                                    # === 측면 광고 필터 (핵심!) ===
                                    # 번호판이 차량 중앙에서 35% 이상 벗어나면 측면 광고로 판단
                                    if abs(p_cx - c_cx) > (c_w * 0.35):
                                        continue
                                    
                                    # 번호판이 너무 크면 오탐지 (차량 너비의 45% 이하만 허용)
                                    if p_w > (c_w * 0.45):
                                        continue
                                    
                                    # === 상단 필터 ===
                                    # 버스/트럭(cls 5,7): 상단 25%만 무시 (앞번호판 허용)
                                    if c_cls in [5, 7]:
                                        if p_cy < (cy1 + c_h * 0.25): continue
                                    else:
                                        # 승용차: 상단 15%만 무시
                                        if p_cy < (cy1 + c_h * 0.15): continue
                                    
                                    valid_loc = True
                                    break
                            
                            # 차량 없어도 이전에 추적하던거면 유지
                            if not valid_loc and track_id != -1 and track_id in plate_memory: valid_loc = True
                             
                            # 차량 없어도 특징이 명확하면 허용
                            if not valid_loc and len(car_boxes) == 0:
                                p_ratio = p_w / (y2-y1) if (y2-y1) > 0 else 0
                                if 2.0 <= p_ratio <= 4.0 and p_cy > frame_height * 0.3: valid_loc = True
                            
                            if not valid_loc: continue

                            # 스무딩 및 속도 계산
                            current_coords = (x1, y1, x2, y2)
                            velocity = (0, 0)
                            if track_id != -1 and track_id in plate_memory:
                                old_info = plate_memory[track_id]
                                old_coords = old_info.get('coords', current_coords)
                                velocity = (
                                    (x1+x2)/2 - (old_coords[0]+old_coords[2])/2,
                                    (y1+y2)/2 - (old_coords[1]+old_coords[3])/2
                                )
                                current_coords = smooth_coordinates(current_coords, old_coords)
                                x1, y1, x2, y2 = current_coords

                            plate_count += 1
                            
                            # 블러 영역 확장
                            bx1, by1, bx2, by2 = get_expanded_blur_region(current_coords, velocity, frame_width, frame_height)
                            
                            # 블러 적용 (안전 커널 적용)
                            roi = frame[by1:by2, bx1:bx2]
                            if roi.size > 0:
                                try:
                                    kw = max(3, (int((bx2-bx1)/1.5) | 1))
                                    kh = max(3, (int((by2-by1)/1.5) | 1))
                                    frame[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (kw, kh), 0)
                                except: pass
                            
                            # 메모리 업데이트 (life=90: 3초@30fps 동안 기억 유지)
                            if track_id != -1:
                                plate_memory[track_id] = {
                                    'coords': (bx1, by1, bx2, by2),
                                    'velocity': velocity,
                                    'life': 90,  # 3초(30fps 기준) 동안 기억 유지
                                    'last_seen': frame_idx,
                                    'confidence_streak': plate_memory.get(track_id, {}).get('confidence_streak', 0) + 1
                                }
                                current_frame_ids.append(track_id)

            # 4. 놓친 번호판 처리 (예측 블러)
            keys_to_remove = []
            for tid, info in plate_memory.items():
                if tid not in current_frame_ids:
                    frames_since = frame_idx - info.get('last_seen', frame_idx)
                    if frames_since > 15: # 너무 오래됨
                        keys_to_remove.append(tid)
                        continue
                    
                    # 위치 예측
                    pred_coords = predict_position(info['coords'], info.get('velocity', (0,0)), min(frames_since, 3))
                    lx1, ly1, lx2, ly2 = pred_coords
                    lx1, ly1 = max(0, lx1), max(0, ly1)
                    lx2, ly2 = min(frame_width, lx2), min(frame_height, ly2)
                    
                    if lx2 <= lx1 or ly2 <= ly1: 
                        keys_to_remove.append(tid)
                        continue
                        
                    # 블러 적용 (안전 커널)
                    roi = frame[ly1:ly2, lx1:lx2]
                    if roi.size > 0:
                        try:
                            kw = max(3, (int((lx2-lx1)/1.5) | 1))
                            kh = max(3, (int((ly2-ly1)/1.5) | 1))
                            frame[ly1:ly2, lx1:lx2] = cv2.GaussianBlur(roi, (kw, kh), 0)
                        except: pass
                    
                    info['life'] -= 1
                    if info['life'] <= 0: keys_to_remove.append(tid)
            
            for k in keys_to_remove: del plate_memory[k]

            # 5. 멀리 있는 차량 확대 분석 (Zoom-in)
            for c_data in car_boxes:
                coords, cls_id = c_data
                cx1, cy1, cx2, cy2 = map(int, coords)
                cw, ch = cx2-cx1, cy2-cy1
                
                pad_w, pad_h = int(cw*0.2), int(ch*0.2)
                bx1, by1 = max(0, cx1-pad_w), max(0, cy1-pad_h)
                bx2, by2 = min(frame_width, cx2+pad_w), min(frame_height, cy2+pad_h)
                
                car_crop = frame[by1:by2, bx1:bx2]
                if car_crop.size == 0: continue
                
                # 확대
                try:
                    input_crop = car_crop
                    if car_crop.shape[1] < 200:
                        input_crop = cv2.resize(car_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                except: continue
                
                # 확대 분석 (conf=0.10으로 민감하게)
                zoom_results = plate_model.predict(input_crop, conf=0.10, imgsz=640, verbose=False)
                if zoom_results:
                    for zr in zoom_results:
                        if zr.boxes:
                            for zb in zr.boxes:
                                zx1, zy1, zx2, zy2 = map(int, zb.xyxy[0].cpu().numpy())
                                
                                # 좌표 복원 (확대한 경우 비율에 맞춰 다시 줄임)
                                if car_crop.shape[1] < 200:
                                    zx1, zy1, zx2, zy2 = int(zx1/3), int(zy1/3), int(zx2/3), int(zy2/3)
                                
                                # 원래 프레임 좌표로 변환
                                gx1, gy1 = bx1 + zx1, by1 + zy1
                                gx2, gy2 = bx1 + zx2, by1 + zy2
                                
                                gx1, gy1 = max(0, gx1), max(0, gy1)
                                gx2, gy2 = min(frame_width, gx2), min(frame_height, gy2)
                                
                                if not is_valid_plate(gx1, gy1, gx2, gy2, frame_width, frame_height): continue
                                
                                # === 확대 분석 위치 검증 (이전 버전) ===
                                p_cy = (gy1 + gy2) / 2
                                p_cx = (gx1 + gx2) / 2
                                
                                # 차종별 상단 필터
                                # 버스/트럭은 상단 70% 무시, 승용차는 상단 30% 무시
                                limit_ratio = 0.70 if cls_id in [5, 7] else 0.30
                                if (p_cy - cy1) < (ch * limit_ratio): continue
                                
                                # 너비 필터: 차량 너비의 50% 이상이면 제외 (거대 블러 방지)
                                if (gx2 - gx1) > (cw * 0.50): continue
                                
                                # 확대 블러
                                roi = frame[gy1:gy2, gx1:gx2]
                                if roi.size > 0:
                                    try:
                                        kw = max(3, (int((gx2-gx1)/2) | 1))
                                        kh = max(3, (int((gy2-gy1)/2) | 1))
                                        frame[gy1:gy2, gx1:gx2] = cv2.GaussianBlur(roi, (kw, kh), 0)
                                        plate_count += 1
                                    except: pass

            out.write(frame)

        cap.release()
        out.release()
        
        # ffmpeg로 AVI → MP4 (H.264) 변환 (브라우저 호환)
        print(f"\n🔄 ffmpeg로 MP4 변환 중...")
        import subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', temp_filepath,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            blurred_filepath
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            # 임시 AVI 파일 삭제
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            print(f"✅ MP4 변환 완료: {blurred_filename}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ ffmpeg 변환 실패, AVI 파일 사용: {e}")
            # 변환 실패시 AVI 파일을 결과로 사용
            blurred_filepath = temp_filepath
            blurred_filename = temp_filename
        
        print(f"분석 완료. 저장됨: {blurred_filename} (Faces: {face_count}, Plates: {plate_count})")

        return {
            "analyzed_video_url": f"/uploads/{blurred_filename}",
            "analyzed_video_path": blurred_filepath,
            "stats": {"faces": face_count, "plates": plate_count},
            "stopped": False
        }

    except Exception as e:
        print("분석 중 오류 발생:")
        traceback.print_exc()
        return {"error": str(e)}

# -----------------------------------------------
# --- API 엔드포인트 ---
# -----------------------------------------------

# 🚨 [신규] 다중 영상 순차 처리용 래퍼 함수
async def run_sequential_analysis(post_id: int, file_paths: List[str]):
    print(f"🚀 다중 영상 분석 스레드 시작 (총 {len(file_paths)}개 파일)")
    
    # DB 세션 새로 생성 (스레드 내부)
    async with async_session() as session:
        statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
        result = await session.execute(statement)
        db_post = result.scalars().one_or_none()
        
        if not db_post: return

        # 기존 결과 리스트 로드
        try:
            current_results = json.loads(db_post.analyzed_video_path)
            if not isinstance(current_results, list): current_results = []
        except:
            current_results = []

        is_stopped = False

        for path in file_paths:
            # 동기 함수를 스레드에서 실행
            result = await asyncio.to_thread(process_video_for_privacy, path, post_id)
            
            if result.get("stopped"):
                print("🛑 사용자에 의해 분석이 중단되었습니다.")
                is_stopped = True
                break
            
            if "error" not in result:
                # 결과 추가
                current_results.append(result['analyzed_video_url'])
                # 중간 저장 (선택 사항: 분석 하나 끝날 때마다 DB 업데이트하려면 여기서 수행)
                
        # 최종 DB 업데이트
        if is_stopped:
            db_post.status = STATUS_PENDING # 다시 대기 상태로
        else:
            db_post.analyzed_video_path = json.dumps(current_results)
            db_post.status = STATUS_COMPLETED
        
        # 중지 시그널 초기화
        if post_id in STOP_SIGNALS:
            del STOP_SIGNALS[post_id]

        session.add(db_post)
        await session.commit()
        print("✅ 모든 분석 작업 종료 및 DB 저장 완료.")


@app.get("/api/download/{file_name}")
async def download_file(file_name: str = Path(..., description="다운로드할 파일의 이름")):
    base_name = os.path.basename(file_name)
    file_path = os.path.join(UPLOAD_DIRECTORY, base_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    return FileResponse(file_path, filename=base_name, media_type='application/octet-stream')

# ... (이하 기존 API 엔드포인트들은 원본 main.py와 동일하 합니다)


@app.get("/api/posts", response_model=PaginatedPostResponse)
async def get_posts(
    session: AsyncSession = Depends(get_async_session),
    search: str = "", 
    page: int = 1, 
    status_filter: Optional[str] = None 
):
    limit: int = 10 
    statement = select(models.AnalysisRequest)
    
    if status_filter == STATUS_PENDING or status_filter == STATUS_IN_PROGRESS: 
        statement = statement.where(or_(models.AnalysisRequest.status == STATUS_PENDING, models.AnalysisRequest.status == STATUS_IN_PROGRESS))
    elif status_filter == STATUS_COMPLETED:
        statement = statement.where(models.AnalysisRequest.status == STATUS_COMPLETED)
    
    if search:
        search_term = f"%{search}%"
        statement = statement.where(or_(models.AnalysisRequest.author.like(search_term), models.AnalysisRequest.email.like(search_term)))

    count_statement = select(func.count()).select_from(statement.subquery())
    total_posts_result = await session.execute(count_statement)
    total_posts = total_posts_result.scalar_one_or_none() or 0
    
    offset = (page - 1) * limit
    statement = statement.order_by(models.AnalysisRequest.id.desc()).offset(offset).limit(limit)
    
    results = await session.execute(statement)
    posts = results.scalars().all()
    
    total_pages = (total_posts + limit - 1) // limit if limit > 0 else 0
    if total_pages == 0 and total_posts > 0: total_pages = 1

    return {"total_posts": total_posts, "total_pages": total_pages, "posts": posts}

@app.get("/api/posts/{post_id}", response_model=PostDetailResponse)
async def get_post_detail(
    post_id: int = Path(...), 
    session: AsyncSession = Depends(get_async_session)
):
    statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()
    if not db_post: raise HTTPException(status_code=404, detail=f"게시글 ID {post_id}를 찾을 수 없습니다.")
    return db_post

@app.post("/api/posts/{post_id}/verify")
async def verify_post_password(
    post_id: int = Path(...),
    password_data: PasswordCheck = Body(...),
    session: AsyncSession = Depends(get_async_session)
):
    statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()
    if not db_post: raise HTTPException(status_code=404, detail=f"게시글 ID {post_id}를 찾을 수 없습니다.")
    if not security.verify_password(password_data.password, db_post.password_hash):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")
    return {"status": "success", "message": "비밀번호가 확인되었습니다."}

# 🚨 [신규] 게시글 수정 API
@app.put("/api/posts/{post_id}")
async def update_post(
    post_id: int = Path(...),
    update_data: PostUpdate = Body(...),
    session: AsyncSession = Depends(get_async_session)
):
    statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()
    
    if not db_post:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")
        
    # 비밀번호 확인
    if not security.verify_password(update_data.password, db_post.password_hash):
        raise HTTPException(status_code=401, detail="비밀번호가 일치하지 않습니다.")
    
    # 수정 가능한 필드만 업데이트
    if update_data.title: db_post.title = update_data.title
    if update_data.content: db_post.content = update_data.content
    if update_data.target_address: db_post.target_address = update_data.target_address
    
    session.add(db_post)
    await session.commit()
    return {"status": "success", "message": "수정되었습니다."}

@app.delete("/api/posts/{post_id}")
async def delete_post(
    post_id: int = Path(...),
    session: AsyncSession = Depends(get_async_session)
):
    statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()
    if not db_post: raise HTTPException(status_code=404, detail="Not Found")

    # 관련 파일 삭제 로직 강화 (JSON 리스트 파싱)
    try:
        # 결과 영상들 삭제
        if db_post.analyzed_video_path:
            try:
                paths = json.loads(db_post.analyzed_video_path)
                if isinstance(paths, list):
                    for p in paths:
                        fname = os.path.basename(p)
                        real_path = os.path.join(UPLOAD_DIRECTORY, fname)
                        if os.path.exists(real_path): os.remove(real_path)
                else: # 예전 방식 (문자열) 처리
                    fname = os.path.basename(db_post.analyzed_video_path)
                    real_path = os.path.join(UPLOAD_DIRECTORY, fname)
                    if os.path.exists(real_path): os.remove(real_path)
            except: pass

        # 원본 영상들 삭제
        if db_post.original_video_path:
            try:
                paths = json.loads(db_post.original_video_path)
                if isinstance(paths, list):
                    for p in paths:
                        if os.path.exists(p): os.remove(p)
                else:
                    if os.path.exists(db_post.original_video_path): os.remove(db_post.original_video_path)
            except: pass
            
    except Exception as e:
        print(f"파일 삭제 중 오류: {e}")

    await session.delete(db_post)
    await session.commit()
    return {"status": "success", "message": "삭제 완료"}

# 🚨 [업데이트] 분석 시작 API (다중 파일 지원)
@app.post("/admin/analyze/{post_id}", response_model=PostDetailResponse) 
async def start_analysis(
    post_id: int = Path(..., description="분석할 게시글 ID"),
    videos: List[UploadFile] = File(..., description="관리자가 업로드하는 원본 영상들"), 
    session: AsyncSession = Depends(get_async_session)
):
    statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()
    if not db_post: raise HTTPException(status_code=404, detail="Not Found")

    # 중지 시그널 초기화
    if post_id in STOP_SIGNALS: del STOP_SIGNALS[post_id]

    db_post.status = STATUS_IN_PROGRESS
    session.add(db_post)
    await session.commit()

    saved_paths = []
    saved_filenames = []

    # 기존 파일 목록 유지 (추가 업로드인 경우)
    try:
        if db_post.original_video_path and db_post.original_video_path != "[]":
            existing_paths = json.loads(db_post.original_video_path)
            if isinstance(existing_paths, list): saved_paths.extend(existing_paths)
            
        if db_post.original_video_filename and db_post.original_video_filename != "[]":
            existing_names = json.loads(db_post.original_video_filename)
            if isinstance(existing_names, list): saved_filenames.extend(existing_names)
    except: pass

    try:
        # 모든 파일 저장
        for video in videos:
            unique_filename = f"{str(uuid.uuid4())}_{video.filename}"
            save_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)
            
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            
            saved_paths.append(save_path)
            saved_filenames.append(video.filename)
            video.file.close()
        
        # DB에 파일 목록 업데이트 (JSON)
        db_post.original_video_path = json.dumps(saved_paths)
        db_post.original_video_filename = json.dumps(saved_filenames)
        session.add(db_post)
        await session.commit()

        # 🚨 비동기 백그라운드 작업 시작 (새로 올린 파일들만 분석하려면 videos로 반복해야 하지만, 편의상 saved_paths 전체 혹은 새로 올린것만 선택 가능. 여기선 간단히 새로 올린 것만 분석하도록 로직 구성)
        # 로직: 방금 저장한 경로들만 분석 리스트에 넣음
        new_file_paths = saved_paths[-len(videos):]
        
        asyncio.create_task(run_sequential_analysis(post_id, new_file_paths))
        
    except Exception as e:
        db_post.status = STATUS_PENDING
        session.add(db_post)
        await session.commit()
        print(f"처리 오류 상세: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"처리 오류: {str(e)}")

    return db_post

# 🚨 [신규] 분석 중지 API
@app.post("/admin/stop/{post_id}")
async def stop_analysis(post_id: int = Path(...)):
    STOP_SIGNALS[post_id] = True
    print(f"🛑 Post {post_id}에 대한 중지 신호 설정됨.")
    return {"status": "stopping", "message": "분석 중지 신호를 보냈습니다."}

# 🚨 [신규] 완료된 영상 삭제 API
@app.delete("/admin/videos/{post_id}")
async def delete_analyzed_video(
    post_id: int = Path(...),
    video_url: str = Body(..., embed=True), # {"video_url": "..."}
    session: AsyncSession = Depends(get_async_session)
):
    statement = select(models.AnalysisRequest).where(models.AnalysisRequest.id == post_id)
    result = await session.execute(statement)
    db_post = result.scalars().one_or_none()
    if not db_post: raise HTTPException(status_code=404, detail="Not Found")
    
    try:
        current_videos = json.loads(db_post.analyzed_video_path)
        if video_url in current_videos:
            current_videos.remove(video_url)
            # 실제 파일 삭제
            fname = os.path.basename(video_url)
            real_path = os.path.join(UPLOAD_DIRECTORY, fname)
            if os.path.exists(real_path): os.remove(real_path)
            
            db_post.analyzed_video_path = json.dumps(current_videos)
            
            # 영상이 하나도 없으면 상태를 다시 PENDING? 아니면 COMPLETED 유지?
            # 사용자 편의를 위해 영상 없으면 PENDING으로 돌림 (선택사항)
            if not current_videos:
                db_post.status = STATUS_PENDING
                
            session.add(db_post)
            await session.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"삭제 실패: {e}")
        
    return {"status": "success", "remaining_videos": current_videos}

@app.post("/request-analysis/", response_model=PostResponse)
async def request_analysis(
    session: AsyncSession = Depends(get_async_session),
    
    title: str = Form(...),
    author: str = Form(...),
    content: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    target_address: str = Form(...), 
):
    hashed_password = security.get_password_hash(password)

    new_request = models.AnalysisRequest(
        title=title,
        author=author,
        content=content,
        email=email,
        password_hash=hashed_password,
        target_address=target_address, 
        status=STATUS_PENDING # 영문    
    )

    try:
        session.add(new_request)
        await session.commit()
        await session.refresh(new_request) 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 저장 실패: {str(e)}")

    return new_request
