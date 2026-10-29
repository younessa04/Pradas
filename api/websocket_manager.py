"""
PRADAS WebSocket Manager.

Manages multiple WebSocket client connections with bounded queues.
Prevents slow clients from blocking the video processing pipeline.
"""

import asyncio
import json
import time
from collections import deque
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

from utils.logging_config import get_logger

logger = get_logger("websocket_manager")


class WebSocketManager:
    """
    Asynchronous WebSocket connection manager.

    Features:
        - Multiple concurrent clients.
        - Bounded per-client message queues.
        - Graceful disconnect handling.
        - Non-blocking broadcast (won't freeze pipeline).
        - Reconnection support (clients can reconnect).
    """

    def __init__(self, max_queue_size: int = 100):
        """
        Args:
            max_queue_size: Maximum queued messages per client.
                Oldest messages are dropped when exceeded.
        """
        self._clients: Dict[str, WebSocket] = {}
        self._queues: Dict[str, deque] = {}
        self._max_queue_size = max_queue_size
        self._client_counter = 0
        self._lock = asyncio.Lock()
        logger.info(f"WebSocketManager initialized: max_queue={max_queue_size}")

    async def connect(self, websocket: WebSocket) -> str:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance.

        Returns:
            Client ID string.
        """
        await websocket.accept()
        async with self._lock:
            self._client_counter += 1
            client_id = f"client_{self._client_counter}"
            self._clients[client_id] = websocket
            self._queues[client_id] = deque(maxlen=self._max_queue_size)

        logger.info(f"WebSocket client connected: {client_id} "
                    f"(total: {self.client_count})")

        # Send welcome message
        try:
            await websocket.send_json({
                "event_type": "connection_established",
                "client_id": client_id,
                "schema_version": "1.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
        except Exception:
            pass

        return client_id

    async def disconnect(self, client_id: str) -> None:
        """
        Remove a disconnected client.

        Args:
            client_id: Client ID to remove.
        """
        async with self._lock:
            self._clients.pop(client_id, None)
            self._queues.pop(client_id, None)
        logger.info(f"WebSocket client disconnected: {client_id} "
                    f"(remaining: {self.client_count})")

    async def broadcast(self, message: dict) -> int:
        """
        Broadcast a message to all connected clients.

        Handles individual client failures without affecting other clients.

        Args:
            message: JSON-serializable message dict.

        Returns:
            Number of clients that received the message.
        """
        if not self._clients:
            return 0

        sent_count = 0
        failed_clients = []

        # Get snapshot of current clients
        async with self._lock:
            clients_snapshot = dict(self._clients)

        message_str = json.dumps(message, default=str)

        for client_id, websocket in clients_snapshot.items():
            try:
                await asyncio.wait_for(
                    websocket.send_text(message_str),
                    timeout=2.0  # Don't wait more than 2s per client
                )
                sent_count += 1
            except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError) as e:
                logger.debug(f"Failed to send to {client_id}: {e}")
                failed_clients.append(client_id)
            except Exception as e:
                logger.warning(f"Unexpected error sending to {client_id}: {e}")
                failed_clients.append(client_id)

        # Clean up failed clients
        for client_id in failed_clients:
            await self.disconnect(client_id)

        return sent_count

    def enqueue_message(self, message: dict) -> None:
        """
        Enqueue a message for all clients (non-async, pipeline-safe).

        This can be called from the synchronous video processing pipeline
        without blocking. Messages are stored in bounded queues.

        Args:
            message: JSON-serializable message dict.
        """
        for client_id, queue in self._queues.items():
            queue.append(message)

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    @property
    def has_clients(self) -> bool:
        """Whether any clients are connected."""
        return len(self._clients) > 0
