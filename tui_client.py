import json
import asyncio
import httpx
from typing import List, Dict, Optional

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static, LoadingIndicator
from textual.containers import Vertical, VerticalScroll
from textual.reactive import reactive

SERVER_URL = "http://localhost:8000"

class ChatBubble(Static):
    """A simple chat bubble widget."""
    def __init__(
        self,
        role: str,
        content: str="",
        is_pending: bool=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.is_pending = is_pending
    
    def on_mount(self) -> None:
        self._update_classes()
        self._refresh_content()

    def _update_classes(self) -> None:
        self.remove_class(
            "role-user",
            "role-assistant", 
            "role-error",
            "pending-bubble"
            )
        if self.role == "user":
            self.add_class("role-user")
        elif self.role == "assistant":
            self.add_class("role-assistant")
        elif self.role == "error":
            self.add_class("role-error")
        
        if self.is_pending:
            self.add_class("pending-bubble")
    
    def set_pending(self, is_pending: bool) -> None:
        self.is_pending = is_pending
        self._update_classes()
    
    def update_text(self, new_content: str) -> None:
        self.content = new_content
        self._refresh_content()
    
    def _refresh_content(self) -> None:
        prefix = "User: " if self.role == "user" else ("Assistant: " if self.role == "assistant" else "Error: ")
        pending_indicator = "[Queued...]" if self.is_pending else ""
        self.update(f"[b]{prefix}{pending_indicator}[/b]\n\n{self.content}")

class TinyInfraTUI(App):
    """A Textual TUI for interacting with the Tiny Infra server."""
    CSS = """
    Screen {
        background: #0f172a;
        layout: vertical;
    }

    #chat-container {
        height: 1fr;
        padding: 1 2;
        overflow-y: scroll;
    }

    #input-dock {
        dock: bottom;
        height: auto;
        padding: 1 2;
        background: #1e293b;
        border-top: solid #334155;
    }

    Input {
        width: 100%;
        border: tall #3b82f6;
        background: #0f172a;
        color: #f8fafc;
    }

    Input:focus {
        border: tall #60a5fa;
    }

    ChatBubble {
        margin: 1 0;
        padding: 1 2;
        max-width: 85%;
    }

    .role-user {
        background: #1e3a8a;
        color: #eff6ff;
        border: round #2563eb;
        align-horizontal: right;
    }

    .role-assistant {
        background: #1e293b;
        color: #f8fafc;
        border: round #475569;
        align-horizontal: left;
    }

    .role-error {
        background: #7f1d1d;
        color: #fee2e2;
        border: round #dc2626;
        align-horizontal: left;
    }

    .pending-bubble {
        opacity: 50%;
    }

    LoadingIndicator {
        height: 3;
        width: 12;
        margin: 1 2;
        align-horizontal: left;
        color: #60a5fa;
    }
    """
    is_generating: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.messages: List[Dict[str, str]] = []
        self.pending_queue: List[tuple[str, ChatBubble]] = []
        self.http_client: Optional[httpx.AsyncClient] = None
        self.current_stream_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="chat-container"):
            yield Vertical(id="chat-messages")
        with Vertical(id="input-dock"):
            yield Input(placeholder="Type your message here...", id="user-input")
        yield Footer()
    
    async def on_mount(self) -> None:
        self.http_client = httpx.AsyncClient(base_url=SERVER_URL, timeout=60.0)
        self.query_one(Input).focus()
    
    async def on_unmount(self) -> None:
        if self.current_stream_task and not self.current_stream_task.done():
            self.current_stream_task.cancel()
        if self.http_client:
            await self.http_client.aclose()
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        
        input_widget = self.query_one(Input)
        input_widget.value = ""

        if text.lower() == "/clear":
            self.messages.clear()
            self.pending_queue.clear()
            chat_container = self.query_one("#chat-container")
            await chat_container.remove_children()
            await chat_container.mount(Static("[dim] Chat history cleared. [/dim]"))
            return
        
        chat_container = self.query_one("#chat-container")

        if self.is_generating:

            user_bubble = ChatBubble(role="user", content=text, is_pending=True)
            await chat_container.mount(user_bubble)
            user_bubble.scroll_visible()
            self.pending_queue.append((text, user_bubble))
        else:
            user_bubble = ChatBubble(role="user", content=text, is_pending=False)
            await chat_container.mount(user_bubble)
            user_bubble.scroll_visible()
            self.current_stream_task = asyncio.create_task(self._process_generation(text, user_bubble))

    async def _process_generation(self, user_text:str,user_bubble: ChatBubble) -> None:
        self.is_generating = True
        user_bubble.set_pending(False)
        self.messages.append({"role": "user", "content": user_text})

        chat_container = self.query_one("#chat-container")
        indicator = LoadingIndicator()
        await chat_container.mount(indicator)
        indicator.scroll_visible()

        assistant_bubble: Optional[ChatBubble] = None
        accumulated_text = []

        try:
            payload = {
                "messages": self.messages,
                "max_new_tokens":1024,
                "temperature":0.7,
                "top_p":0.9,
                "top_k":20,
                "stream": True
            }

            async with self.http_client.stream("POST", "/generate", json=payload) as response:
                if response.status_code != 200:
                    err_content = await response.aread()
                    raise RuntimeError(f"Server returned status code {response.status_code}: {err_content.decode()}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        raw_json = line[6:].strip()
                        if not raw_json:
                            continue
                        data = json.loads(raw_json)

                        if data.get("error"):
                            raise RuntimeError(data["error"])
                        
                        chunk = data.get("text", "")
                        if chunk:
                            accumulated_text.append(chunk)
                            if assistant_bubble is None:
                                await indicator.remove()
                                assistant_bubble = ChatBubble(role="assistant", content="".join(accumulated_text))
                                await chat_container.mount(assistant_bubble)
                            else:
                                assistant_bubble.update_text("".join(accumulated_text))
                            assistant_bubble.scroll_visible()
                        if data.get("done"):
                            break
                full_reply = "".join(accumulated_text)
                if full_reply:
                    self.messages.append({"role": "assistant", "content": full_reply})
        except asyncio.CancelledError:
            if indicator in chat_container.children:
                await indicator.remove()
            if assistant_bubble:
                assistant_bubble.update_text("".join(accumulated_text) + "\n\n[dim italic][Stream Cancelled][/dim italic]")
                raise
        except Exception as e:
            if indicator in container.children:
                await indicator.remove()

            error_bubble = ChatBubble(role="error", content=f"Failed to generate response: {e}")
            await container.mount(error_bubble)
            error_bubble.scroll_visible()
            self.messages.pop()
        finally:
            self.is_generating = False

            if self.pending_queue:
                next_text, next_user_bubble = self.pending_queue.pop(0)
                self.current_stream_task = asyncio.create_task(self._process_generation(next_text,next_user_bubble))

if __name__ == "__main__":
    app = TinyInfraTUI()
    app.run()