"""Tests for conversation history manager."""

import pytest

from app.utils.history import HistoryManager, Conversation, Message


class TestConversation:
    """Test suite for Conversation dataclass."""

    def test_add_message(self):
        """Verify messages are added correctly."""
        conv = Conversation(max_turns=3)

        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi there!")

        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Hello"

    def test_truncates_old_messages(self):
        """Verify old messages are removed when exceeding max turns."""
        conv = Conversation(max_turns=2)  # 4 messages max

        conv.add_message("user", "msg1")
        conv.add_message("assistant", "resp1")
        conv.add_message("user", "msg2")
        conv.add_message("assistant", "resp2")
        conv.add_message("user", "msg3")  # This should cause truncation

        assert len(conv.messages) == 4
        # First message should be "msg2" (msg1 and resp1 evicted)
        assert conv.messages[0].content == "resp1"

    def test_get_history_returns_openai_format(self):
        """Verify history is returned in OpenAI message format."""
        conv = Conversation()

        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")

        history = conv.get_history()

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi!"}

    def test_clear_removes_all_messages(self):
        """Verify clear removes all messages."""
        conv = Conversation()

        conv.add_message("user", "Hello")
        conv.add_message("assistant", "Hi!")
        conv.clear()

        assert len(conv.messages) == 0


class TestHistoryManager:
    """Test suite for HistoryManager."""

    def test_add_user_message(self):
        """Verify user messages are added correctly."""
        manager = HistoryManager(max_turns=3)

        manager.add_user_message(123, "Hello")

        history = manager.get_history(123)
        assert len(history) == 1
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        """Verify assistant messages are added correctly."""
        manager = HistoryManager(max_turns=3)

        manager.add_assistant_message(123, "Hi there!")

        history = manager.get_history(123)
        assert len(history) == 1
        assert history[0]["role"] == "assistant"

    def test_separate_histories_per_user(self):
        """Verify each user has separate history."""
        manager = HistoryManager()

        manager.add_user_message(123, "Hello from user 123")
        manager.add_user_message(456, "Hello from user 456")

        history_123 = manager.get_history(123)
        history_456 = manager.get_history(456)

        assert len(history_123) == 1
        assert len(history_456) == 1
        assert history_123[0]["content"] == "Hello from user 123"
        assert history_456[0]["content"] == "Hello from user 456"

    def test_clear_history_for_user(self):
        """Verify clearing history for a specific user."""
        manager = HistoryManager()

        manager.add_user_message(123, "Hello")
        manager.add_user_message(456, "Hi")
        manager.clear_history(123)

        assert len(manager.get_history(123)) == 0
        assert len(manager.get_history(456)) == 1

    def test_clear_all_histories(self):
        """Verify clearing all histories."""
        manager = HistoryManager()

        manager.add_user_message(123, "Hello")
        manager.add_user_message(456, "Hi")
        manager.clear_all()

        assert manager.get_user_count() == 0

    def test_get_empty_history_for_new_user(self):
        """Verify new users get empty history."""
        manager = HistoryManager()

        history = manager.get_history(999)

        assert history == []

    def test_respects_max_turns(self):
        """Verify history respects max_turns setting."""
        manager = HistoryManager(max_turns=2)

        for i in range(5):
            manager.add_user_message(123, f"msg{i}")
            manager.add_assistant_message(123, f"resp{i}")

        history = manager.get_history(123)

        # Should only have last 2 turns (4 messages)
        assert len(history) == 4


class TestHistoryManagerStats:
    """Test suite for history manager statistics."""

    def test_get_stats(self):
        """Verify stats returns expected information."""
        manager = HistoryManager(max_turns=3)

        manager.add_user_message(123, "Hello")
        manager.add_assistant_message(123, "Hi")
        manager.add_user_message(456, "Hey")

        stats = manager.get_stats()

        assert stats["user_count"] == 2
        assert stats["total_messages"] == 3
        assert stats["max_turns"] == 3

    def test_user_count(self):
        """Verify user count is accurate."""
        manager = HistoryManager()

        manager.add_user_message(1, "a")
        manager.add_user_message(2, "b")
        manager.add_user_message(3, "c")

        assert manager.get_user_count() == 3
