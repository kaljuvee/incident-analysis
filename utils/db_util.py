import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, Text, String, func, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
from sqlalchemy import func
from sqlalchemy import inspect
from sqlalchemy.sql import text

# Load environment variables from .env file
load_dotenv()

# Database connection
SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI")
if not SQLALCHEMY_DATABASE_URI:
    raise ValueError("SQLALCHEMY_DATABASE_URI environment variable is not set")

engine = create_engine(SQLALCHEMY_DATABASE_URI)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    data_set_id = Column(String(36), index=True)  # UUID is 36 characters long
    document_content = Column(Text)
    created_at = Column(DateTime, default=func.now())  # Add this line

def create_tables():
    Base.metadata.create_all(engine)
    add_created_at_column()

def store_documents(contents, data_set_id=None):
    if data_set_id is None:
        data_set_id = str(uuid.uuid4())
    session = Session()
    try:
        for content in contents:
            doc = Document(document_content=content, data_set_id=data_set_id)
            session.add(doc)
        session.commit()
        return data_set_id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def read_documents_by_dataset(data_set_id):
    session = Session()
    try:
        docs = session.query(Document).filter(Document.data_set_id == data_set_id).all()
        return [doc.document_content for doc in docs]
    finally:
        session.close()

def get_distinct_dataset_ids():
    session = Session()
    try:
        distinct_ids = session.query(Document.data_set_id).distinct().all()
        return [id[0] for id in distinct_ids]
    finally:
        session.close()

def get_datasets_with_counts():
    session = Session()
    try:
        datasets_with_counts = session.query(
            Document.data_set_id,
            func.count(Document.id).label('doc_count'),
            func.min(Document.created_at).label('created_at')
        ).group_by(Document.data_set_id).all()
        return [(dataset_id, count, created_at) for dataset_id, count, created_at in datasets_with_counts]
    finally:
        session.close()

def add_created_at_column():
    from sqlalchemy import Column, DateTime
    from sqlalchemy.sql import func
    
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    inspector = inspect(engine)
    
    if 'created_at' not in [c['name'] for c in inspector.get_columns('documents')]:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE documents ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
