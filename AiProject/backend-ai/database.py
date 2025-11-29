# database.py

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
from typing import AsyncGenerator

# 1. .env 파일에서 환경 변수(DATABASE_URL)를 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL이 .env 파일에 설정되지 않았습니다.")

# 2. 비동기(asyncio) DB 연결 엔진 생성
#    FastAPI는 비동기 프레임워크이므로, DB 연결도 비동기로 설정하는 것이 좋습니다.
async_engine = create_async_engine(DATABASE_URL, echo=True)

# 3. 비동기 세션(DB와의 대화 창구)을 만드는 함수
async_session = sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# 4. (중요) DB 테이블을 생성하는 함수
async def create_db_and_tables():
    async with async_engine.begin() as conn:
        # SQLModel의 모든 모델(테이블)을 생성
        await conn.run_sync(SQLModel.metadata.create_all)

# 5. API 엔드포인트에서 DB 세션을 쉽게 사용할 수 있게 도와주는 함수
# 이 함수는 AsyncSession을 yield(생성)하는 AsyncGenerator입니다 라는 뜻
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session