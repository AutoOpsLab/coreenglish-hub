from typing import Optional
from sqlmodel import SQLModel, Field, create_engine

# ----------------------------------------
# Database models
# ----------------------------------------

class Unit(SQLModel, table=True):
    """
    A Unit represents a collection of lessons generated for a specific
    topic, grade, and genre.
    """
    id: str = Field(primary_key=True, description="UUID of the unit")
    topic: str = Field(..., description="Topic of the unit")
    grade: int = Field(..., description="Grade level")
    genre: str = Field(..., description="Genre for lessons")


class Lesson(SQLModel, table=True):
    """
    A Lesson represents either a summary or detailed lesson plan
    associated with a Unit.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    unit_id: str = Field(foreign_key="unit.id", description="The unit UUID")
    idx: int = Field(..., description="Lesson index within the unit")
    lesson_type: str = Field(..., description="Category: reading, writing, etc.")
    title: str = Field(..., description="Lesson title")
    objective: str = Field(..., description="One-sentence learning objective")

    # Detailed fields (JSON-encoded strings)
    standards: str = Field(default="[]", description="JSON-encoded list of standards")
    success_criteria: str = Field(default="{}", description="JSON-encoded dict of success criteria")
    ell_strategies: str = Field(default="[]", description="JSON-encoded list of ELL strategies")
    activities: str = Field(default="[]", description="JSON-encoded list of activities")
    assessment: str = Field(default="", description="Assessment description")


# ----------------------------------------
# Engine and initialization
# ----------------------------------------

# Use synchronous SQLite driver to avoid greenlet issues
DATABASE_URL = "sqlite:///./coreenglish.db"
engine = create_engine(
    DATABASE_URL,
    echo=True,
    connect_args={"check_same_thread": False}
)

def init_db() -> None:
    """
    Create database tables.
    """
    SQLModel.metadata.create_all(engine)