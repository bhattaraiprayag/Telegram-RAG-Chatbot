"""Conversation history manager for Telegram users."""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Message:
    """A single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Conversation:
    """Conversation history for a single user."""

    messages: list[Message] = field(default_factory=list)
    max_turns: int = 3

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation.

        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        self.messages.append(Message(role=role, content=content))

        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_history(self) -> list[dict[str, str]]:
        """
        Get conversation history in OpenAI message format.

        Returns:
            List of dicts with 'role' and 'content'
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class HistoryManager:
    """
    Manages conversation history for multiple users.

    NOTE: This is an in-memory implementation. History is lost on restart.
    For production, consider persisting to Redis or a database.
    """

    def __init__(self, max_turns: int = 3) -> None:
        """
        Initialize history manager.

        Args:
            max_turns: Maximum conversation turns to maintain per user
        """
        self.max_turns = max_turns
        self._conversations: dict[int, Conversation] = defaultdict(
            lambda: Conversation(max_turns=self.max_turns)
        )

    def add_user_message(self, user_id: int, content: str) -> None:
        """
        Add a user message to the conversation.

        Args:
            user_id: Telegram user ID
            content: Message content
        """
        self._conversations[user_id].add_message("user", content)

    def add_assistant_message(self, user_id: int, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Args:
            user_id: Telegram user ID
            content: Message content
        """
        self._conversations[user_id].add_message("assistant", content)

    def get_history(self, user_id: int) -> list[dict[str, str]]:
        """
        Get conversation history for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            List of message dicts in OpenAI format
        """
        return self._conversations[user_id].get_history()

    def clear_history(self, user_id: int) -> None:
        """
        Clear conversation history for a user.

        Args:
            user_id: Telegram user ID
        """
        self._conversations[user_id].clear()

    def clear_all(self) -> None:
        """Clear all conversation histories."""
        self._conversations.clear()

    def get_user_count(self) -> int:
        """Get number of users with active conversations."""
        return len(self._conversations)

    def get_stats(self) -> dict[str, int]:
        """
        Get history manager statistics.

        Returns:
            Dict with user_count and total_messages
        """
        total_messages = sum(
            len(conv.messages) for conv in self._conversations.values()
        )
        return {
            "user_count": self.get_user_count(),
            "total_messages": total_messages,
            "max_turns": self.max_turns,
        }
