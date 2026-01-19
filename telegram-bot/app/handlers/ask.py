"""Ask command handler for RAG queries."""

import logging

from telegram import Update
from telegram.error import BadRequest
from telegram.ext import ContextTypes

from ..rag.orchestrator import RAGOrchestrator
from ..utils.history import HistoryManager


logger = logging.getLogger(__name__)


async def safe_reply(message, text: str) -> None:
    """
    Safely send a reply, falling back to plain text if Markdown parsing fails.

    Args:
        message: Telegram message object
        text: Text to send
    """
    try:
        await message.reply_text(text, parse_mode="Markdown")
    except BadRequest as e:
        if "can't parse entities" in str(e).lower():
            logger.warning(f"Markdown parsing failed, sending as plain text: {e}")
            await message.reply_text(text)
        else:
            raise


async def ask_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    orchestrator: RAGOrchestrator,
    history_manager: HistoryManager,
) -> None:
    """
    Handle /ask command for RAG queries.

    Args:
        update: Telegram update object
        context: Bot context
        orchestrator: RAG orchestrator instance
        history_manager: Conversation history manager
    """
    if update.message is None or update.effective_user is None:
        return

    # Extract query from command
    if context.args:
        query = " ".join(context.args)
    else:
        await update.message.reply_text(
            "Please provide a question. Usage: /ask <your question>",
        )
        return

    user_id = update.effective_user.id

    # Get conversation history
    chat_history = history_manager.get_history(user_id)

    # Send "typing" indicator
    await update.message.chat.send_action("typing")

    # Stream response from RAG
    response_parts = []
    try:
        async for token in orchestrator.query(query, chat_history=chat_history):
            response_parts.append(token)
    except Exception as e:
        logger.exception(f"RAG query failed: {e}")
        await update.message.reply_text(f"❌ An error occurred: {str(e)}")
        return

    response = "".join(response_parts)

    # Update conversation history
    history_manager.add_user_message(user_id, query)
    history_manager.add_assistant_message(user_id, response)

    # Send response (split if too long)
    if len(response) > 4000:
        # Telegram message limit is 4096 characters
        for i in range(0, len(response), 4000):
            chunk = response[i : i + 4000]
            await safe_reply(update.message, chunk)
    else:
        await safe_reply(update.message, response)


async def clear_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    history_manager: HistoryManager,
) -> None:
    """
    Handle /clear command to clear conversation history.

    Args:
        update: Telegram update object
        context: Bot context
        history_manager: Conversation history manager
    """
    if update.message is None or update.effective_user is None:
        return

    user_id = update.effective_user.id
    history_manager.clear_history(user_id)

    await update.message.reply_text(
        "✅ Your conversation history has been cleared.",
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    orchestrator: RAGOrchestrator,
    history_manager: HistoryManager,
) -> None:
    """
    Handle /stats command to show system statistics.

    Args:
        update: Telegram update object
        context: Bot context
        orchestrator: RAG orchestrator instance
        history_manager: Conversation history manager
    """
    if update.message is None:
        return

    cache_stats = orchestrator.get_cache_stats()
    history_stats = history_manager.get_stats()

    stats_text = f"""📊 System Statistics

Embedding Cache:
• Size: {cache_stats['size']} / {cache_stats['max_size']}
• Hit Rate: {cache_stats['hit_rate']:.1%}
• Hits: {cache_stats['hits']} | Misses: {cache_stats['misses']}

Conversation History:
• Active Users: {history_stats['user_count']}
• Total Messages: {history_stats['total_messages']}
• Max Turns: {history_stats['max_turns']}
"""

    await update.message.reply_text(stats_text)
