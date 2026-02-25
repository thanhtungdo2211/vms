from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:admin123@192.168.6.229:5432/iotvms_new"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
# print("--", SessionLocal)