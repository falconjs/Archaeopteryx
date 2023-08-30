from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    and_,
    false,
    func,
    inspect,
    or_,
    text,
)

from sqlalchemy import and_
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy.orm import relationship, joinedload
from sqlalchemy.orm import Session

from sqlalchemy.orm import deferred

Base = declarative_base()

class TaskInstance(Base):
    __tablename__ = "task_instance"

    task_id = Column(String(30), primary_key=True, nullable=False)
    dag_id = Column(String(30), primary_key=True, nullable=False)
    run_id = Column(String(30), primary_key=True, nullable=False)
    comments = Column(String(30))

    __table_args__ = (
        Index("ti_dag_run", dag_id, run_id),
        PrimaryKeyConstraint(
            "dag_id", "task_id", "run_id", name="task_instance_pkey", mssql_clustered=True
        ),
        ForeignKeyConstraint(
            [dag_id, run_id],
            ["dag_run.dag_id", "dag_run.run_id"],
            name="task_instance_dag_run_fkey",
            ondelete="CASCADE",
        ),
    )

    dag_run = relationship("DagRun", back_populates="task_instances", lazy="joined", innerjoin=True)
    execution_date = association_proxy("dag_run", "execution_date")
    
class DagRun(Base):
    __tablename__ = "dag_run"

    id = Column(Integer, primary_key=True)
    dag_id = Column(String(30), nullable=False)
    run_id = Column(String(30), nullable=False)
    queued_at = Column(DateTime)
    execution_date = Column(DateTime, nullable=False)

    task_instances = relationship(
        TaskInstance, back_populates="dag_run", cascade="save-update, merge, delete, delete-orphan"
    )

from sqlalchemy.orm import load_only

if __name__ == "__main__":
    engine = create_engine("sqlite:///sqlalchemy.orm.db")
    Base.metadata.create_all(engine)

    session = Session(engine)

    # query = session.query(TaskInstance)
    # query = session.query(TaskInstance).options(load_only(TaskInstance.task_id,), joinedload(TaskInstance.dag_run).load_only(DagRun.execution_date))
    # query = session.query(TaskInstance).options(joinedload(TaskInstance.dag_run).load_only(DagRun.execution_date))
    query = session.query(TaskInstance).options(load_only(TaskInstance.execution_date))
    print(type(query))
    print(query)
