"""
PostgreSQL Database Connection Manager
Reads database configuration from YAML config file.
"""
import os
import yaml
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def _load_db_config() -> str:
    """
    Load database URL from face recognition config.
    Priority:
    1. DATABASE_URL environment variable
    2. Config file database.url
    3. Construct from database.host/port/name/user/password
    
    Returns:
        str: PostgreSQL connection URL
    """
    # Priority 1: Environment variable
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        print(f"[PGDB] Using DATABASE_URL from environment")
        return env_url
    
    # Priority 2: Load from config file
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    db_cfg = config.get("database", {})
    
    # Priority 2a: Use full URL if provided
    if db_cfg.get("url"):
        url = db_cfg["url"]
        print(f"[PGDB] Using database.url from config: {url}")
        return url
    
    # Priority 2b: Construct URL from components
    host = db_cfg.get("host", "localhost")
    port = db_cfg.get("port", 5432)
    db_name = db_cfg.get("name", "iotvms_new")
    user = db_cfg.get("user", "postgres")
    password = db_cfg.get("password", "admin123")
    
    url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    print(f"[PGDB] Constructed database URL from config components")
    return url


# Initialize database connection
DATABASE_URL = _load_db_config()
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,         # Connection pool size
    max_overflow=10,     # Max overflow connections
    echo=False           # Don't log SQL queries (set True for debugging)
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

print(f"[PGDB] Database engine initialized: {engine.url.host}:{engine.url.port}/{engine.url.database}")