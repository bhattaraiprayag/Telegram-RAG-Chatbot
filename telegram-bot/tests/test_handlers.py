"""Tests for Telegram command handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.handlers.help import help_command, start_command, HELP_TEXT
from app.handlers.ask import ask_command, clear_command, stats_command
from app.utils.history import HistoryManager
from app.rag.cache import EmbeddingCache


@pytest.fixture
def mock_update():
    """Create mock Telegram Update object."""
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    update.message.chat = MagicMock()
    update.message.chat.send_action = AsyncMock()
    update.effective_user = MagicMock()
    update.effective_user.id = 12345
    return update


@pytest.fixture
def mock_context():
    """Create mock Telegram Context object."""
    context = MagicMock()
    context.args = []
    return context


@pytest.fixture
def mock_orchestrator():
    """Create mock RAG orchestrator."""
    orchestrator = MagicMock()

    async def mock_query(*args, **kwargs):
        yield "Test response"
        yield "\n\n---\n**Sources:**\n"
        yield "- [1] test.md > Section\n"

    orchestrator.query = mock_query
    orchestrator.get_cache_stats.return_value = {
        "size": 10,
        "max_size": 100,
        "hits": 5,
        "misses": 3,
        "hit_rate": 0.625,
    }
    return orchestrator


@pytest.fixture
def history_manager():
    """Create history manager for testing."""
    return HistoryManager(max_turns=3)


class TestHelpCommand:
    """Test suite for help command."""

    @pytest.mark.asyncio
    async def test_help_command_sends_help_text(self, mock_update, mock_context):
        """Verify help command sends help text."""
        await help_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "DocuRAG Bot" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_help_command_handles_no_message(self, mock_context):
        """Verify help command handles missing message gracefully."""
        update = MagicMock()
        update.message = None

        # Should not raise
        await help_command(update, mock_context)

    @pytest.mark.asyncio
    async def test_start_command_sends_welcome(self, mock_update, mock_context):
        """Verify start command sends welcome message."""
        await start_command(mock_update, mock_context)

        mock_update.message.reply_text.assert_called_once()
        call_args = mock_update.message.reply_text.call_args
        assert "Welcome" in call_args[0][0]


class TestAskCommand:
    """Test suite for ask command."""

    @pytest.mark.asyncio
    async def test_ask_command_requires_query(
        self, mock_update, mock_context, mock_orchestrator, history_manager
    ):
        """Verify ask command requires a query."""
        mock_context.args = []

        await ask_command(
            mock_update, mock_context, mock_orchestrator, history_manager
        )

        call_args = mock_update.message.reply_text.call_args
        assert "Please provide a question" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_ask_command_sends_response(
        self, mock_update, mock_context, mock_orchestrator, history_manager
    ):
        """Verify ask command sends RAG response."""
        mock_context.args = ["What", "is", "the", "policy"]

        await ask_command(
            mock_update, mock_context, mock_orchestrator, history_manager
        )

        # Should have sent typing action
        mock_update.message.chat.send_action.assert_called_with("typing")

        # Should have sent response
        mock_update.message.reply_text.assert_called()

    @pytest.mark.asyncio
    async def test_ask_command_updates_history(
        self, mock_update, mock_context, mock_orchestrator, history_manager
    ):
        """Verify ask command updates conversation history."""
        mock_context.args = ["test", "query"]
        user_id = mock_update.effective_user.id

        await ask_command(
            mock_update, mock_context, mock_orchestrator, history_manager
        )

        history = history_manager.get_history(user_id)
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert "test query" in history[0]["content"]


class TestClearCommand:
    """Test suite for clear command."""

    @pytest.mark.asyncio
    async def test_clear_command_clears_history(
        self, mock_update, mock_context, history_manager
    ):
        """Verify clear command clears user history."""
        user_id = mock_update.effective_user.id
        history_manager.add_user_message(user_id, "test")

        await clear_command(mock_update, mock_context, history_manager)

        assert len(history_manager.get_history(user_id)) == 0

    @pytest.mark.asyncio
    async def test_clear_command_sends_confirmation(
        self, mock_update, mock_context, history_manager
    ):
        """Verify clear command sends confirmation message."""
        await clear_command(mock_update, mock_context, history_manager)

        call_args = mock_update.message.reply_text.call_args
        assert "cleared" in call_args[0][0].lower()


class TestStatsCommand:
    """Test suite for stats command."""

    @pytest.mark.asyncio
    async def test_stats_command_shows_cache_stats(
        self, mock_update, mock_context, mock_orchestrator, history_manager
    ):
        """Verify stats command shows cache statistics."""
        await stats_command(
            mock_update, mock_context, mock_orchestrator, history_manager
        )

        call_args = mock_update.message.reply_text.call_args
        response = call_args[0][0]
        assert "Embedding Cache" in response
        assert "Hit Rate" in response

    @pytest.mark.asyncio
    async def test_stats_command_shows_history_stats(
        self, mock_update, mock_context, mock_orchestrator, history_manager
    ):
        """Verify stats command shows history statistics."""
        history_manager.add_user_message(123, "test")

        await stats_command(
            mock_update, mock_context, mock_orchestrator, history_manager
        )

        call_args = mock_update.message.reply_text.call_args
        response = call_args[0][0]
        assert "Conversation History" in response
        assert "Active Users" in response
