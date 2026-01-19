"""Telegram RAG Bot - Main Application Entry Point."""

import asyncio
import hashlib
import logging
import os
from functools import partial
from pathlib import Path

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import settings
from .chunking import ChunkingEngine
from .database import QdrantDB
from .handlers.ask import ask_command, clear_command, stats_command
from .handlers.help import help_command, start_command
from .handlers.upload import handle_document, list_files_command
from .rag.orchestrator import RAGOrchestrator
from .services.ml_api_client import MLAPIClient
from .utils.history import HistoryManager


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


SAMPLE_DOCS_DIR = Path("./sample_docs")


class BotApplication:
    """Main bot application class."""

    def __init__(self) -> None:
        """Initialize bot application components."""
        self.db = QdrantDB()
        self.ml_client = MLAPIClient()
        self.chunking_engine = ChunkingEngine()
        self.orchestrator = RAGOrchestrator(
            db=self.db,
            ml_client=self.ml_client,
        )
        self.history_manager = HistoryManager(
            max_turns=settings.max_history_turns
        )

    async def seed_sample_documents(self) -> None:
        """
        Seed sample documents on first startup.

        Ingests documents from sample_docs/ directory if they haven't
        been indexed yet.
        """
        if not SAMPLE_DOCS_DIR.exists():
            logger.info("No sample_docs directory found, skipping seeding")
            return

        sample_files = list(SAMPLE_DOCS_DIR.glob("*.md")) + \
                       list(SAMPLE_DOCS_DIR.glob("*.txt"))

        if not sample_files:
            logger.info("No sample documents found to seed")
            return

        logger.info(f"Found {len(sample_files)} sample documents")

        from markitdown import MarkItDown
        md_converter = MarkItDown()

        for file_path in sample_files:
            try:
                # Calculate file hash
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                # Skip if already indexed
                if self.db.file_exists(file_hash):
                    logger.info(f"Skipping already indexed: {file_path.name}")
                    continue

                logger.info(f"Indexing sample document: {file_path.name}")

                # Convert to markdown
                result = md_converter.convert(str(file_path))
                markdown_content = result.text_content

                if not markdown_content or not markdown_content.strip():
                    logger.warning(f"Could not extract text from {file_path.name}")
                    continue

                # Chunk document
                parents, children = self.chunking_engine.chunk_document(
                    markdown_content, file_hash, file_path.name
                )

                if not children:
                    logger.warning(f"No chunks produced for {file_path.name}")
                    continue

                # Embed chunks
                chunk_texts = [c.content for c in children]
                embeddings = await self.ml_client.embed(chunk_texts, is_query=False)

                # Store in database
                chunk_dicts = [
                    {
                        "id": c.id,
                        "content": c.content,
                        "parent_id": c.parent_id,
                        "file_hash": c.file_hash,
                        "file_name": c.file_name,
                        "chunk_index": c.chunk_index,
                        "header_path": c.header_path,
                    }
                    for c in children
                ]

                parent_dicts = [
                    {
                        "id": p.id,
                        "content": p.content,
                        "file_hash": p.file_hash,
                        "file_name": p.file_name,
                        "header_path": p.header_path,
                        "child_ids": p.child_ids,
                    }
                    for p in parents
                ]

                self.db.store_chunks(
                    chunk_dicts,
                    embeddings["dense_vecs"],
                    embeddings["sparse_vecs"],
                )
                self.db.store_parents(parent_dicts)

                logger.info(
                    f"Indexed {file_path.name}: "
                    f"{len(parents)} parents, {len(children)} chunks"
                )

            except Exception as e:
                logger.error(f"Error indexing {file_path.name}: {e}")

    def create_application(self) -> Application:
        """
        Create and configure the Telegram application.

        Returns:
            Configured Application instance
        """
        if not settings.telegram_token:
            raise ValueError(
                "TELEGRAM_TOKEN not set. Please configure it in .env"
            )

        app = Application.builder().token(settings.telegram_token).build()

        # Register command handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))

        # Ask command with dependencies injected
        async def ask_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await ask_command(
                update, context, self.orchestrator, self.history_manager
            )

        app.add_handler(CommandHandler("ask", ask_handler))

        # Clear command
        async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await clear_command(update, context, self.history_manager)

        app.add_handler(CommandHandler("clear", clear_handler))

        # Stats command
        async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await stats_command(
                update, context, self.orchestrator, self.history_manager
            )

        app.add_handler(CommandHandler("stats", stats_handler))

        # Files command
        async def files_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await list_files_command(update, context, self.db)

        app.add_handler(CommandHandler("files", files_handler))

        # Document upload handler
        async def document_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
            await handle_document(
                update, context, self.db, self.ml_client, self.chunking_engine
            )

        app.add_handler(MessageHandler(filters.Document.ALL, document_handler))

        return app

    async def run(self) -> None:
        """Run the bot application."""
        logger.info("Starting DocuRAG Telegram Bot...")

        # Seed sample documents
        try:
            await self.seed_sample_documents()
        except Exception as e:
            logger.error(f"Error seeding sample documents: {e}")
            logger.info("Continuing without sample documents...")

        # Create and run application
        app = self.create_application()

        logger.info("Bot is running! Press Ctrl+C to stop.")

        await app.initialize()
        await app.start()
        await app.updater.start_polling(drop_pending_updates=True)

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("Shutting down...")
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            await self.orchestrator.close()


def main() -> None:
    """Main entry point."""
    bot = BotApplication()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
