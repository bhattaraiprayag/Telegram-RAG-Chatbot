"""File upload handler for document ingestion."""

import hashlib
import os
from pathlib import Path

from markitdown import MarkItDown
from telegram import Update
from telegram.ext import ContextTypes

from ..chunking import ChunkingEngine
from ..database import QdrantDB
from ..services.ml_api_client import MLAPIClient

UPLOAD_DIR = Path("./uploads")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".doc", ".epub"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    db: QdrantDB,
    ml_client: MLAPIClient,
    chunking_engine: ChunkingEngine,
) -> None:
    """
    Handle document upload for ingestion.

    Args:
        update: Telegram update object
        context: Bot context
        db: Qdrant database client
        ml_client: ML API client for embeddings
        chunking_engine: Document chunking engine
    """
    if update.message is None or update.message.document is None:
        return

    document = update.message.document

    # Validate file extension
    file_name = document.file_name or "unknown"
    file_ext = Path(file_name).suffix.lower()

    if file_ext not in ALLOWED_EXTENSIONS:
        await update.message.reply_text(
            f"❌ Unsupported file type: `{file_ext}`\n"
            f"Supported: {', '.join(ALLOWED_EXTENSIONS)}",
            parse_mode="Markdown",
        )
        return

    # Validate file size
    if document.file_size and document.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ File too large. Maximum size is {MAX_FILE_SIZE // 1024 // 1024} MB.",
            parse_mode="Markdown",
        )
        return

    # Send progress update
    progress_msg = await update.message.reply_text(
        "📥 Downloading file...",
        parse_mode="Markdown",
    )

    try:
        # Download file
        file = await context.bot.get_file(document.file_id)

        UPLOAD_DIR.mkdir(exist_ok=True)
        file_path = UPLOAD_DIR / file_name

        await file.download_to_drive(str(file_path))

        # Calculate file hash
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Check if already indexed
        if db.file_exists(file_hash):
            await progress_msg.edit_text(
                f"ℹ️ This file has already been indexed: `{file_name}`",
                parse_mode="Markdown",
            )
            os.remove(file_path)
            return

        # Convert to markdown
        await progress_msg.edit_text(
            "📄 Converting document to text...",
            parse_mode="Markdown",
        )

        md = MarkItDown()
        result = md.convert(str(file_path))
        markdown_content = result.text_content

        if not markdown_content or not markdown_content.strip():
            await progress_msg.edit_text(
                "❌ Could not extract text from the document.",
                parse_mode="Markdown",
            )
            os.remove(file_path)
            return

        # Chunk document
        await progress_msg.edit_text(
            "✂️ Chunking document...",
            parse_mode="Markdown",
        )

        parents, children = chunking_engine.chunk_document(
            markdown_content, file_hash, file_name
        )

        if not children:
            await progress_msg.edit_text(
                "❌ Document produced no chunks. It may be too short.",
                parse_mode="Markdown",
            )
            os.remove(file_path)
            return

        # Embed chunks
        await progress_msg.edit_text(
            f"🧮 Embedding {len(children)} chunks...",
            parse_mode="Markdown",
        )

        chunk_texts = [c.content for c in children]
        embeddings = await ml_client.embed(chunk_texts, is_query=False)

        # Store in database
        await progress_msg.edit_text(
            "💾 Storing in database...",
            parse_mode="Markdown",
        )

        # Prepare chunk dicts
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

        db.store_chunks(
            chunk_dicts,
            embeddings["dense_vecs"],
            embeddings["sparse_vecs"],
        )
        db.store_parents(parent_dicts)

        # Success
        await progress_msg.edit_text(
            f"✅ **Document indexed successfully!**\n\n"
            f"📄 File: `{file_name}`\n"
            f"📦 Parents: {len(parents)}\n"
            f"🧩 Chunks: {len(children)}\n\n"
            f"You can now ask questions about this document using `/ask <question>`",
            parse_mode="Markdown",
        )

        # Cleanup uploaded file (optional - keep for debugging)
        # os.remove(file_path)

    except Exception as e:
        await progress_msg.edit_text(
            f"❌ Error processing document: {str(e)}",
            parse_mode="Markdown",
        )
        raise


async def list_files_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    db: QdrantDB,
) -> None:
    """
    Handle /files command to list indexed documents.

    Args:
        update: Telegram update object
        context: Bot context
        db: Qdrant database client
    """
    if update.message is None:
        return

    files = db.get_all_files()

    if not files:
        await update.message.reply_text(
            "📂 No documents have been indexed yet.\nUpload a document to get started!",
            parse_mode="Markdown",
        )
        return

    file_list = "\n".join(f"• `{f['file_name']}`" for f in files)

    await update.message.reply_text(
        f"📂 **Indexed Documents ({len(files)}):**\n\n{file_list}",
        parse_mode="Markdown",
    )
