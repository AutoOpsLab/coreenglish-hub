# backend/db.py
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship, create_engine

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str

    units: List["Unit"] = Relationship(back_populates="owner")

class Unit(SQLModel, table=True):
    id: str = Field(primary_key=True)
    topic: str
    grade: int
    genre: str

    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")
    owner: Optional[User] = Relationship(back_populates="units")

class Lesson(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    unit_id: str = Field(foreign_key="unit.id")
    idx: int
    lesson_type: str
    title: str
    objective: str
    standards: str = Field(default="[]")
    success_criteria: str = Field(default="{}")
    ell_strategies: str = Field(default="[]")
    activities: str = Field(default="[]")
    assessment: str = Field(default="")

DATABASE_URL = "sqlite:///./coreenglish.db"
engine = create_engine(
    DATABASE_URL, echo=False, connect_args={"check_same_thread": False}
)

def init_db() -> None:
    SQLModel.metadata.create_all(engine)
