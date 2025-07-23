"""
QCM Generator Pro - Database Connection and Session Management

This module provides database connection management, session handling,
and database utilities using SQLAlchemy with async support.
"""

import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from ..models.database import Base
from .config import settings
from .exceptions import DatabaseConnectionError, DatabaseError

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Database connection and session manager."""

    def __init__(self, database_url: str | None = None):
        """Initialize database manager with connection URL."""
        self.database_url = database_url or settings.database.url
        self._engine: Engine | None = None
        self._async_engine = None
        self._session_maker: sessionmaker | None = None
        self._async_session_maker = None

    @property
    def engine(self) -> Engine:
        """Get or create synchronous database engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @property
    def async_engine(self):
        """Get or create asynchronous database engine."""
        if self._async_engine is None:
            # Convert sync URL to async URL for SQLite
            async_url = self.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            self._async_engine = create_async_engine(
                async_url,
                echo=settings.database.echo,
                pool_pre_ping=True,
            )
        return self._async_engine

    @property
    def session_maker(self) -> sessionmaker:
        """Get session maker for synchronous sessions."""
        if self._session_maker is None:
            self._session_maker = sessionmaker(
                bind=self.engine,
                class_=Session,
                expire_on_commit=False,
            )
        return self._session_maker

    @property
    def async_session_maker(self):
        """Get session maker for asynchronous sessions."""
        if self._async_session_maker is None:
            self._async_session_maker = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._async_session_maker

    def _create_engine(self) -> Engine:
        """Create database engine with appropriate settings."""
        try:
            if self.database_url.startswith("sqlite"):
                # SQLite-specific configuration
                engine = create_engine(
                    self.database_url,
                    echo=settings.database.echo,
                    poolclass=StaticPool,
                    pool_pre_ping=True,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 30,
                    },
                )
                # Enable foreign key constraints for SQLite
                @event.listens_for(engine, "connect")
                def set_sqlite_pragma(dbapi_connection, connection_record):
                    cursor = dbapi_connection.cursor()
                    cursor.execute("PRAGMA foreign_keys=ON")
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                    cursor.execute("PRAGMA temp_store=memory")
                    cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                    cursor.close()

            else:
                # PostgreSQL or other database configuration
                engine = create_engine(
                    self.database_url,
                    echo=settings.database.echo,
                    poolclass=QueuePool,
                    pool_size=settings.database.pool_size,
                    pool_pre_ping=settings.database.pool_pre_ping,
                    pool_recycle=settings.database.pool_recycle,
                    connect_args={
                        "options": "-c timezone=utc"
                    } if "postgresql" in self.database_url else {},
                )

            logger.info(f"Database engine created: {self.database_url}")
            return engine

        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            raise DatabaseConnectionError(
                "Failed to create database connection",
                details={"database_url": self.database_url, "error": str(e)},
                cause=e
            )

    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise DatabaseError(
                "Failed to create database tables",
                details={"error": str(e)},
                cause=e
            )

    async def create_tables_async(self) -> None:
        """Create all database tables asynchronously."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully (async)")
        except Exception as e:
            logger.error(f"Failed to create database tables (async): {e}")
            raise DatabaseError(
                "Failed to create database tables",
                details={"error": str(e)},
                cause=e
            )

    def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise DatabaseError(
                "Failed to drop database tables",
                details={"error": str(e)},
                cause=e
            )

    async def drop_tables_async(self) -> None:
        """Drop all database tables asynchronously."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully (async)")
        except Exception as e:
            logger.error(f"Failed to drop database tables (async): {e}")
            raise DatabaseError(
                "Failed to drop database tables",
                details={"error": str(e)},
                cause=e
            )

    def reset_database(self) -> None:
        """Reset database by dropping and recreating all tables."""
        logger.info("Resetting database...")
        self.drop_tables()
        self.create_tables()
        logger.info("Database reset complete")

    async def reset_database_async(self) -> None:
        """Reset database asynchronously by dropping and recreating all tables."""
        logger.info("Resetting database (async)...")
        await self.drop_tables_async()
        await self.create_tables_async()
        logger.info("Database reset complete (async)")

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    async def test_connection_async(self) -> bool:
        """Test database connection asynchronously."""
        try:
            async with self.async_engine.begin() as conn:
                await conn.execute("SELECT 1")
            logger.info("Database connection test successful (async)")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed (async): {e}")
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.session_maker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(
                "Database session error",
                details={"error": str(e)},
                cause=e
            )
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic cleanup."""
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error (async): {e}")
                raise DatabaseError(
                    "Database session error",
                    details={"error": str(e)},
                    cause=e
                )

    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")
        if self._async_engine:
            # Note: async engine disposal should be done with await in async context
            logger.info("Async database engine should be disposed in async context")

    async def close_async(self) -> None:
        """Close database connections asynchronously."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database engine disposed")


# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for getting sessions
get_db_session = db_manager.get_session
get_async_db_session = db_manager.get_async_session


def init_database() -> None:
    """Initialize the database with tables."""
    logger.info("Initializing database...")
    db_manager.create_tables()
    logger.info("Database initialization complete")


async def init_database_async() -> None:
    """Initialize the database with tables asynchronously."""
    logger.info("Initializing database (async)...")
    await db_manager.create_tables_async()
    logger.info("Database initialization complete (async)")


def reset_database() -> None:
    """Reset the database."""
    db_manager.reset_database()


async def reset_database_async() -> None:
    """Reset the database asynchronously."""
    await db_manager.reset_database_async()


# Database dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency to get database session."""
    with db_manager.get_session() as session:
        yield session


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency to get async database session."""
    async with db_manager.get_async_session() as session:
        yield session


# Health check function
def check_database_health() -> dict:
    """Check database health for monitoring."""
    try:
        is_healthy = db_manager.test_connection()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database_url": db_manager.database_url.split("@")[-1] if "@" in db_manager.database_url else "sqlite",
            "engine_pool_size": getattr(db_manager.engine.pool, "size", None),
            "engine_pool_checked_out": getattr(db_manager.engine.pool, "checkedout", None),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


async def check_database_health_async() -> dict:
    """Check database health asynchronously for monitoring."""
    try:
        is_healthy = await db_manager.test_connection_async()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "database_url": db_manager.database_url.split("@")[-1] if "@" in db_manager.database_url else "sqlite",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
