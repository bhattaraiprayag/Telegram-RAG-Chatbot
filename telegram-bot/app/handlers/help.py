"""Help command handler."""

from telegram import Update
from telegram.ext import ContextTypes

HELP_TEXT = """🤖 **DocuRAG Bot - Document Q&A Assistant**

I can answer questions based on documents that have been uploaded to me.

**Commands:**
• `/ask <question>` - Ask a question about the uploaded documents
• `/summarize <file>` - Get a concise summary of the provided file/document
• `/files` - List all indexed documents
• `/clear` - Clear your conversation history
• `/stats` - Show system statistics
• `/help` - Show this help message

**File Upload:**
Simply send me a document (PDF, TXT, MD, DOCX) and I'll process it for you.

**Tips:**
• Be specific with your questions
• I'll cite sources in my answers
• I remember our last 3 conversation turns

---
*Powered by RAG (Retrieval-Augmented Generation)*
"""


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /help command.

    Args:
        update: Telegram update object
        context: Bot context
    """
    if update.message is None:
        return

    await update.message.reply_text(
        HELP_TEXT,
        parse_mode="Markdown",
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /start command.

    Args:
        update: Telegram update object
        context: Bot context
    """
    if update.message is None:
        return

    welcome_text = """👋 **Welcome to DocuRAG Bot!**

I'm your intelligent document assistant. I can answer questions based on documents you upload.

To get started:
1. Upload a document (PDF, TXT, MD, DOCX)
2. Ask questions using `/ask <your question>`

Type `/help` for more information.
"""

    await update.message.reply_text(
        welcome_text,
        parse_mode="Markdown",
    )
