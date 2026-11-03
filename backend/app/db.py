from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def make_engine(url=None):
    """Create a SQLAlchemy engine; SQLite needs one extra flag for use across threads."""
    url = url or settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
