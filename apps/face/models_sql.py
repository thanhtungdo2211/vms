from sqlalchemy import Column, Integer, String, ForeignKey, JSON, DateTime
from sqlalchemy.orm import relationship, declarative_base, sessionmaker, joinedload
#from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from datetime import datetime
from apps.face.pg_database import SessionLocal
import json

Base = declarative_base()


class AccessGroup(Base):
    __tablename__ = 'access_groups'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    ai_face = Column(Integer, nullable=False)
    ai_uniform = Column(Integer, nullable=False)

    users = relationship("AccessUser", back_populates="group")

class AccessUser(Base):
    __tablename__ = 'access_users'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    phone_mail = Column(String, nullable=True)
    avatar = Column(String, nullable=True)
    group_id = Column(Integer, ForeignKey('access_groups.id', name="fk_user_group"), nullable=True)
    status = Column(String, default='Chưa được xác thực', nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # updated_at = Column(DateTime)
    person_id = Column(String, nullable=True, index=True)

    group = relationship("AccessGroup", back_populates="users")
    features = relationship("UserFeature",back_populates="user", cascade="all, delete-orphan", passive_deletes=True)
    images = relationship(
            "UserImage",
            back_populates="user",
            cascade="all, delete-orphan",
            passive_deletes=True,
            foreign_keys='UserImage.user_id'
        )



class UserImage(Base):
    __tablename__ = 'user_images'

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey('access_users.id', ondelete="CASCADE"), nullable=False)
    image_path = Column(String, nullable=False)

    user = relationship("AccessUser", back_populates="images", foreign_keys=[user_id])

class UserFeature(Base):
    __tablename__ = 'user_features'

    id = Column(Integer, primary_key=True)
    # user_id = Column(String, ForeignKey('access_users.id', name="fk_feature_user"), nullable=False)
    user_id = Column(String, ForeignKey("access_users.id",name="fk_feature_user", ondelete="CASCADE"), nullable=False)
    feature = Column(JSON, nullable=False)

    user = relationship("AccessUser", back_populates="features")


class Stream(Base):
    __tablename__ = 'stream'

    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(Integer, ForeignKey('edge.id'))
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    status = Column(Integer, default=0, nullable=False)

    edge = relationship("Edge", back_populates="streams")

class Edge(Base):
    __tablename__ = 'edge'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    address = Column(String, nullable=False)
    port = Column(Integer, nullable=False)
    status = Column(Integer, default=0, nullable=False)
    serial_id = Column(Integer, ForeignKey('serial.id'))
    num_camera = Column(Integer)
    last_status_update = Column(DateTime, default=datetime.utcnow, nullable=False)

    serial = relationship("Serial", backref="edges")
    streams = relationship("Stream", back_populates="edge")

class Serial(Base):
    __tablename__ = "serial"

    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(100), unique=True, nullable=False)

# class AccessEvent(Base):
#     __tablename__ = "access_events"

#     id = Column(Integer, primary_key=True)
#     user_id = Column(String, ForeignKey("access_users.id"))
#     stream_id = Column(Integer, ForeignKey("stream.id"))
#     edge_id = Column(Integer, ForeignKey("edge.id"))
#     image_face = Column(String)
#     time_access = Column(DateTime, default=datetime.utcnow)
#     # Relationships
#     user = relationship("AccessUser")
#     stream = relationship("Stream")
#     edge = relationship("Edge")

class AccessEvent(Base):
    __tablename__ = 'access_events'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey('access_users.id'), nullable=True)
    time_access = Column(DateTime, default=datetime.utcnow, nullable=False)
    edge_id = Column(Integer, ForeignKey('edge.id'), nullable=False)  # Mối quan hệ với bảng Edge
    stream_id = Column(Integer, ForeignKey('stream.id'), nullable=False)  # Mối quan hệ với bảng Stream
    image_face = Column(String, nullable=True)
    uniform_warning = Column(Integer, nullable=True)
    image_uniform = Column(String, nullable=False)
    image_full = Column(String, nullable=True)
    person_id = Column(String, ForeignKey("access_users.person_id", ondelete="SET NULL"), nullable=True, index=True)


# def get_user_name_by_id(session: SessionLocal, user_id: str) -> str:

#     user = session.query(AccessUser).filter_by(id=user_id).first()
#     return user.name if user else None

# def get_stream_url_by_id_config(idc):
#     config_path = "./deepstream/app/pipelines/configs/config_face.json"
#     with open(config_path, 'r') as f:
#         config = json.load(f)
#         key = "source" + str(idc)
#         if key in config["sources"]:
#             return config["sources"][key]
#     return None
    
# def get_group_id_by_user_id(session: SessionLocal, user_id: str) -> str:

#     user = session.query(AccessUser).filter_by(id=user_id).first()
#     return user.group_id if user else None

# def get_ai_uniform_by_group_id(session: SessionLocal, group_id: str) -> str:

#     ret = session.query(AccessGroup).filter_by(id=group_id).first()
#     return ret.ai_uniform if ret else None




