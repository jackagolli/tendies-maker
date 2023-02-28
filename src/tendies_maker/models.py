import datetime
from typing import List
from typing import Optional

from sqlalchemy import ForeignKey
from sqlalchemy import String, Date
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column


class Base(DeclarativeBase):
    pass


class TrainingData(Base):
    __tablename__ = "training_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime.datetime] = mapped_column(Date())
    name: Mapped[str] = mapped_column(String(30))
    fullname: Mapped[Optional[str]]

    def __repr__(self) -> str:
        return f"TrainingData(id={self.id!r})"
