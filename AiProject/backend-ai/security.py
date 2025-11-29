from passlib.context import CryptContext
import hashlib

# 사용할 암호화 방식(bcrypt)을 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # 사용자가 입력한 비밀번호(plain)과
    # DB에 저장된 비밀번호(hashed)를 비교합니다.

    sha256_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()

    return pwd_context.verify(sha256_hash, hashed_password)

def get_password_hash(password: str) -> str:
    # 비밀번호를 암호화하여 반환합니다.
    sha256_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    return pwd_context.hash(sha256_hash)