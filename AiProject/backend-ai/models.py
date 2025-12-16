# models.py

from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

# 'AnalysisRequest'라는 이름의 테이블을 정의
# SQLModel을 상속받으면, 이것이 DB 테이블이자 API 데이터 모델이 됩니다.
class AnalysisRequest(SQLModel, table=True):
    # (Primary Key) 게시글 ID, 자동으로 1씩 증가
    id: Optional[int] = Field(default=None, primary_key=True) 
    
    # 게시글 제목
    title: str = Field(default="")
    
    # 게시글 내용
    content: Optional[str] = Field(default=None)
    
    # 작성자 이름
    author: str = Field(default="")
    
    # 요청자 이메일, DB에 인덱스(색인)를 만들어 빠르게 찾을 수 있게 함
    email: str = Field(index=True)
    
    # (중요) 암호화된 비밀번호가 저장될 곳
    password_hash: str 
    
    # 대상 주소 (민원 위치)
    target_address: str = Field(default="")
    
    # 원본 영상 파일명 (예: my_video.mp4)
    original_video_filename: Optional[str] = Field(default=None)
    
    # 서버에 저장된 원본 영상 경로 (예: uploads/...)
    original_video_path: Optional[str] = Field(default=None)
    
    # (Nullable) 분석 완료된 영상 경로 (관리자가 나중에 채움)
    analyzed_video_path: Optional[str] = Field(default=None) 
    
    # 현재 상태 (예: PENDING, COMPLETED)
    status: str = Field(default="PENDING") 
    
    # 생성 시간 (자동으로 현재 시간이 기록됨)
    created_at: datetime = Field(default_factory=datetime.utcnow)