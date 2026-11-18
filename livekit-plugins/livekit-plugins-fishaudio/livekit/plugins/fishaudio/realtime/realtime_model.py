from __future__ import annotations

import asyncio
import os
import re
import time
import weakref
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp
import msgpack
import numpy as np

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    llm,
    utils,
)
from livekit.agents.metrics import RealtimeModelMetrics
from livekit.agents.metrics.base import Metadata
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ..log import logger
from ..tts import DEFAULT_BASE_URL, USER_AGENT

OwnSpeech = Literal["natural", "quiet"]

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

_DUPLEX_PATH = "/v1/duplex/converse"
# Caller audio is the session clock: Fish expects a frame every 20 ms, silence included.
_FRAME_MS = 20
_FRAME_BYTES = SAMPLE_RATE * 2 * _FRAME_MS // 1000
# Old caller audio is worse than dropped caller audio.
_MAX_INPUT_BUFFER_BYTES = SAMPLE_RATE * 2
# Fish streams speech at playback pace in bursts, so playout starts behind this much silence.
_JITTER_BUFFER_MS = 200
# After this long without a forwarded frame the playout queue has drained.
_PLAYOUT_IDLE = 0.25
# Sound the voice makes between replies ends after this much silence.
_AMBIENT_SILENCE_MS = 300
# Sound that has to make way for a tool call fades out over this long.
_FADE_OUT_MS = 30
# A reply the caller cut can be picked up again this long after it stopped.
_CUT_RESUME_WINDOW = 6.0
# Fish commits or drops the caller's words about a second after they stop.
_CALLER_COMMIT_WAIT = 1.5
_ASIDE_POLL_INTERVAL = 0.1
_DEFAULT_MAX_TEXT_BYTES = 4096
# The session.start was refused; a new attempt would be refused too.
_INVALID_REQUEST_CLOSE_CODE = 4400
_HAS_WORDS = re.compile(r"\w")


@dataclass
class _RealtimeOptions:
    voice: NotGivenOr[str]
    own_speech: OwnSpeech
    eager_turns: bool
    noise_cancellation: bool
    playback_delay_ms: int
    text_rules: bool
    base_url: str
    api_key: str
    conn_options: APIConnectOptions

    def get_ws_url(self) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{_DUPLEX_PATH}"


@dataclass
class _Reply:
    """One ``reply_id`` on the wire, shown to the framework as one generation."""

    id: str
    message_id: str
    user_initiated: bool
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    created_at: float = field(default_factory=time.time)
    eager_id: str | None = None
    """Set while the reply answers a ``turn.eager`` that Fish has not committed yet."""
    exposed: bool = False
    """The framework has the generation; before that the reply is speculative."""
    cancelled: bool = False
    sent_text: str = ""
    sent_first: bool = False
    sent_last: bool = False
    wire_stopped: bool = False
    """No more text may be sent under this ``reply_id``."""
    unsent_tail: str = ""
    """Text the LLM wrote after the wire stopped, kept in case the reply is resumed."""
    message_sent: bool = False
    started: bool = False
    done: bool = False
    closed: bool = False
    llm_done: bool = False
    pushed_chars: int = 0
    """How much of the heard text the framework has been given."""
    has_audio: bool = False
    first_audio_at: float | None = None
    held_silence: list[bytes] = field(default_factory=list)
    """Silent frames after the last sound, played only if more sound follows."""
    held_calls: list[llm.FunctionCall] = field(default_factory=list)
    """Tool calls waiting for ``turn.committed``; a speculative reply must not run tools."""
    llm_task: asyncio.Task[None] | None = None
    usage: llm.CompletionUsage | None = None
    fut: asyncio.Future[llm.GenerationCreatedEvent] | None = None
    ambient: bool = False
    """Sound the voice made by itself between replies; it has no text."""
    resumed: bool = False
    """The rest of a reply that was cut; it is not picked up a second time."""


@dataclass
class _Cut:
    """A reply the caller stopped, which resumes if their sound turns out not to be a turn."""

    reply: _Reply
    unheard_text: str
    created_at: float


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        llm: llm.LLM | str,
        voice: NotGivenOr[str] = NOT_GIVEN,
        own_speech: OwnSpeech = "natural",
        eager_turns: bool = True,
        noise_cancellation: bool = True,
        playback_delay_ms: int = 300,
        text_rules: bool = True,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """
        Create a Fish Audio Duplex realtime model.

        See https://docs.fish.audio/features/duplex-api for more details on the Duplex API.

        Fish Audio listens, decides when the user's turn starts and ends, and speaks. The
        ``llm`` writes the replies: the model runs it on every user turn and streams its
        text to Fish Audio, so instructions, chat context and tools work as with any LLM.

        Fish Audio handles interruptions itself. It tells a real interruption from a
        backchannel or noise, stops its own speech, and reports the text the user heard.

        Args:
            llm (llm.LLM | str): The LLM that writes the replies. A string names a LiveKit
                Inference model, such as ``"google/gemma-4-31b-it"``.
            voice (NotGivenOr[str]): Fish Audio voice model ID. Uses Fish Audio's default
                voice if unset.
            own_speech (OwnSpeech): ``"natural"`` lets the voice add its own backchannels
                and breaths between replies; ``"quiet"`` turns them off. Defaults to
                ``"natural"``.
            eager_turns (bool): Start the LLM as soon as the user pauses, before Fish Audio
                confirms the turn has ended. Tool calls still wait for the confirmation.
                Defaults to True.
            noise_cancellation (bool): Denoise the user's audio before Fish Audio listens
                to it. Defaults to True.
            playback_delay_ms (int): Delay between Fish Audio sending audio and the user
                hearing it, in milliseconds. Fish Audio uses it to work out what the user
                heard before an interruption. Defaults to 300.
            text_rules (bool): Append Fish Audio's rules for writing spoken text to the
                instructions. Defaults to True.
            api_key (NotGivenOr[str]): Fish Audio API key. Reads ``FISH_API_KEY`` if unset.
            base_url (NotGivenOr[str]): Custom base URL. Defaults to ``https://api.fish.audio``.
            http_session (aiohttp.ClientSession | None): Optional aiohttp session to use.
            conn_options (APIConnectOptions): Retry and timeout settings for the connection.
        """
        super().__init__(
            capabilities=_capabilities(),
        )

        fish_api_key = api_key if is_given(api_key) else os.getenv("FISH_API_KEY")
        if not fish_api_key:
            raise ValueError(
                "Fish Audio API key is required, either as argument or set "
                "FISH_API_KEY environment variable"
            )

        self._owns_llm = isinstance(llm, str)
        self._llm = _resolve_llm(llm)
        self._opts = _RealtimeOptions(
            voice=voice,
            own_speech=own_speech,
            eager_turns=eager_turns,
            noise_cancellation=noise_cancellation,
            playback_delay_ms=playback_delay_ms,
            text_rules=text_rules,
            base_url=base_url if is_given(base_url) else DEFAULT_BASE_URL,
            api_key=fish_api_key,
            conn_options=conn_options,
        )
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return "duplex"

    @property
    def provider(self) -> str:
        return "fishaudio"

    @property
    def llm(self) -> llm.LLM:
        """The LLM that writes the replies."""
        return self._llm

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            self._http_session = utils.http_context.http_session()
        return self._http_session

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        # turn detection is what the Duplex API is; it cannot be turned off
        session = RealtimeSession(self)
        self._sessions.add(session)
        return session

    async def aclose(self) -> None:
        for session in list(self._sessions):
            await session.aclose()
        if self._owns_llm:
            await self._llm.aclose()


def _capabilities() -> llm.RealtimeCapabilities:
    return llm.RealtimeCapabilities(
        # Fish Audio reports the heard text itself
        message_truncation=False,
        turn_detection=True,
        user_transcription=True,
        # the framework asks for the reply to a tool result, so its step limit applies
        auto_tool_reply_generation=False,
        audio_output=True,
        manual_function_calls=True,
        # Fish Audio decides what interrupts the agent
        supports_overlapping_speech=True,
        mutable_chat_context=True,
        mutable_instructions=True,
        mutable_tools=True,
        per_response_tool_choice=True,
        supports_say=True,
    )


def _resolve_llm(model: llm.LLM | str) -> llm.LLM:
    if not isinstance(model, str):
        return model

    # imported lazily: inference pulls in every hosted model type
    from livekit.agents import inference

    return inference.LLM(model)


class RealtimeSession(llm.RealtimeSession[Literal["fishaudio_server_event_received"]]):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = realtime_model._opts
        self._llm = realtime_model._llm

        self._tools = llm.ToolContext.empty()
        self._tool_choice: llm.ToolChoice | None = None
        self._instructions: str | None = None
        self._chat_ctx = llm.ChatContext.empty()
        # items this session added that the framework has not sent back yet
        self._unacked_item_ids: set[str] = set()

        self._connected = False
        # whether the last connection attempt got as far as session.started
        self._was_connected = False
        self._closing = False
        self._session_id: str | None = None
        self._text_rules: dict[str, Any] = {}
        self._max_text_bytes = _DEFAULT_MAX_TEXT_BYTES
        self._send_ch: utils.aio.Chan[dict[str, Any]] | None = None
        # reply frames written before session.started
        self._pending_frames: list[dict[str, Any]] = []
        self._last_error = ""

        self._input_buf = bytearray()
        self._input_resampler: rtc.AudioResampler | None = None

        self._replies: dict[str, _Reply] = {}
        self._speculative: _Reply | None = None
        self._ambient: _Reply | None = None
        self._ambient_silence_ms = 0.0
        # a tool call is waiting for the floor, so the voice's own sound is not played
        self._ambient_muted = False
        self._last_forward_at = 0.0

        self._conn_tag = utils.shortuuid("")
        self._turn_started_at: dict[str, float] = {}
        self._cut: _Cut | None = None
        self._held_asides: list[tuple[str, float]] = []
        self._last_backchannel = ""
        self._caller_heard_at: float | None = None
        self._aside_task: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[Any]] = set()

        self._main_atask = asyncio.create_task(self._main_task(), name="fishaudio-realtime")

    # -- RealtimeSession ---------------------------------------------------------------

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    @property
    def session_id(self) -> str | None:
        """Fish Audio's ID for the current connection, for support requests."""
        return self._session_id

    async def update_instructions(self, instructions: str) -> None:
        self._instructions = instructions

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        incoming = chat_ctx.copy(
            exclude_instructions=True,
            exclude_empty_message=True,
            exclude_handoff=True,
            exclude_config_update=True,
        )
        incoming_ids = set()
        for item in incoming.items:
            incoming_ids.add(item.id)

        # the framework records the agent's speech once it has played, which can be after
        # this update was built: what this session added since stays until it comes back
        kept: list[llm.ChatItem] = []
        for item in self._chat_ctx.items:
            if item.id in incoming_ids:
                self._unacked_item_ids.discard(item.id)
                continue
            if item.id in self._unacked_item_ids:
                kept.append(item)

        self._chat_ctx = incoming
        for item in kept:
            self._chat_ctx.insert(item)

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        self._tools = llm.ToolContext(tools)

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        if is_given(tool_choice):
            self._tool_choice = tool_choice

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closing:
            return

        for resampled in self._resample_audio(frame):
            self._input_buf.extend(resampled.data.tobytes())

        overflow = len(self._input_buf) - _MAX_INPUT_BUFFER_BYTES
        if overflow > 0:
            del self._input_buf[:overflow]

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        if self._closing:
            fut.set_exception(llm.RealtimeError("the session is closed"))
            return fut

        reply = self._new_reply(user_initiated=True)
        reply.fut = fut
        self._expose(reply)
        self._start_llm(
            reply,
            instructions=instructions if is_given(instructions) else None,
            tool_choice=tool_choice,
            tools=tools,
        )
        return fut

    def say(self, text: str | AsyncIterable[str]) -> asyncio.Future[llm.GenerationCreatedEvent]:
        fut: asyncio.Future[llm.GenerationCreatedEvent] = asyncio.Future()
        if self._closing:
            fut.set_exception(llm.RealtimeError("the session is closed"))
            return fut

        reply = self._new_reply(user_initiated=True)
        reply.fut = fut
        self._expose(reply)
        reply.llm_task = self._create_task(self._say_task(reply, text))
        return fut

    def commit_audio(self) -> None:
        pass  # caller audio is consumed continuously, there is no buffer to commit

    def clear_audio(self) -> None:
        pass

    def interrupt(self) -> None:
        # a speculative reply is not the framework's to stop: it has not been told of it
        for reply in list(self._replies.values()):
            if reply.exposed and not reply.done and not reply.ambient:
                self._cancel_reply(reply)
        self._cut = None

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass  # Fish Audio reports the heard text, and only that is kept

    async def aclose(self) -> None:
        if self._closing:
            return
        self._closing = True

        await utils.aio.cancel_and_wait(self._main_atask)
        self._reset_replies("the session closed")
        if self._aside_task is not None:
            await utils.aio.cancel_and_wait(self._aside_task)
        await utils.aio.cancel_and_wait(*self._tasks)

    # -- connection --------------------------------------------------------------------

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        conn_options = self._opts.conn_options
        retries = 0
        while not self._closing:
            try:
                await self._run_connection()
                retries = 0
            except Exception as e:
                retryable = not isinstance(e, APIError) or e.retryable
                recoverable = retryable and retries < conn_options.max_retry
                self._emit_error(e, recoverable=recoverable)
                if not recoverable:
                    self._reset_replies("the connection to Fish Audio failed")
                    return
                retries += 1
                await asyncio.sleep(conn_options.retry_interval)

            if self._closing:
                return

            if self._was_connected:
                # replies in flight died with the connection; the conversation itself is ours
                self._was_connected = False
                self._reset_replies("the connection to Fish Audio was lost")
                self.emit("session_reconnected", llm.RealtimeSessionReconnectedEvent())

    async def _run_connection(self) -> None:
        connect_started_at = time.perf_counter()
        ws = await self._connect_ws()
        self._conn_tag = utils.shortuuid("")
        self._last_error = ""
        send_ch = utils.aio.Chan[dict[str, Any]]()
        self._send_ch = send_ch
        send_ch.send_nowait(self._session_start_frame())

        async def _send_task() -> None:
            async for frame in send_ch:
                await ws.send_bytes(msgpack.packb(frame))

        async def _pump_task() -> None:
            silence = bytes(_FRAME_BYTES)
            loop = asyncio.get_running_loop()
            next_at = loop.time()
            while True:
                if self._connected:
                    chunk = silence
                    if len(self._input_buf) >= _FRAME_BYTES:
                        chunk = bytes(self._input_buf[:_FRAME_BYTES])
                        del self._input_buf[:_FRAME_BYTES]
                    send_ch.send_nowait({"type": "audio", "data": chunk})

                next_at += _FRAME_MS / 1000
                delay = next_at - loop.time()
                if delay < -0.1:
                    # fell behind (a blocked loop): the missed frames are not sent late
                    next_at = loop.time()
                    delay = 0
                await asyncio.sleep(max(delay, 0))

        async def _recv_task() -> None:
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    detail = self._last_error or "no error reported"
                    if ws.close_code == _INVALID_REQUEST_CLOSE_CODE or not self._connected:
                        raise APIStatusError(
                            f"Fish Audio refused the session: {detail}",
                            status_code=ws.close_code or -1,
                            retryable=False,
                        )
                    raise APIConnectionError(f"Fish Audio closed the session: {detail}")

                if msg.type == aiohttp.WSMsgType.ERROR:
                    raise APIConnectionError("Fish Audio connection lost") from ws.exception()

                if msg.type != aiohttp.WSMsgType.BINARY:
                    continue

                event = msgpack.unpackb(msg.data)
                if event.get("type") == "audio":
                    self._on_audio(event["data"])
                    continue

                if event.get("type") == "session.started" and not self._connected:
                    self._report_connection_acquired(time.perf_counter() - connect_started_at)

                self._handle_event(event)

        tasks = [
            asyncio.create_task(_send_task(), name="fishaudio-realtime-send"),
            asyncio.create_task(_pump_task(), name="fishaudio-realtime-pump"),
            asyncio.create_task(_recv_task(), name="fishaudio-realtime-recv"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            self._connected = False
            self._send_ch = None
            send_ch.close()
            await utils.aio.cancel_and_wait(*tasks)
            await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "User-Agent": USER_AGENT,
        }
        try:
            return await asyncio.wait_for(
                self._realtime_model._ensure_http_session().ws_connect(
                    self._opts.get_ws_url(), headers=headers
                ),
                self._opts.conn_options.timeout,
            )
        except aiohttp.WSServerHandshakeError as e:
            error_code = ""
            if e.headers is not None:
                error_code = e.headers.get("x-fish-error-code", "")
            raise APIStatusError(
                f"Fish Audio refused the connection: {e.status} {error_code or e.message}",
                status_code=e.status,
                # a busy server (429) or a server error is worth another attempt
                retryable=e.status == 429 or e.status >= 500,
            ) from e
        except asyncio.TimeoutError as e:
            raise APIConnectionError("Fish Audio connection timed out") from e
        except aiohttp.ClientError as e:
            raise APIConnectionError("Fish Audio connection failed") from e

    def _session_start_frame(self) -> dict[str, Any]:
        frame: dict[str, Any] = {
            "type": "session.start",
            "audio": {
                "encoding": "pcm_s16le",
                "sample_rate": SAMPLE_RATE,
                "noise_cancellation": self._opts.noise_cancellation,
            },
            "turns": {
                "eager": self._opts.eager_turns,
                "backchannel": True,
                "interim": True,
                "speech_started": True,
            },
            "own_speech": self._opts.own_speech,
            "playback": {"delay_ms": self._opts.playback_delay_ms},
        }
        if is_given(self._opts.voice):
            frame["voice"] = {"reference_id": self._opts.voice}
        return frame

    def _send(self, frame: dict[str, Any]) -> None:
        if self._connected and self._send_ch is not None:
            self._send_ch.send_nowait(frame)
            return
        if not self._closing:
            self._pending_frames.append(frame)

    def _emit_error(self, error: Exception, *, recoverable: bool) -> None:
        self.emit(
            "error",
            llm.RealtimeModelError(
                timestamp=time.time(),
                label=self._realtime_model.label,
                error=error,
                recoverable=recoverable,
            ),
        )

    # -- audio -------------------------------------------------------------------------

    def _resample_audio(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        if frame.sample_rate == SAMPLE_RATE and frame.num_channels == NUM_CHANNELS:
            return [frame]

        if self._input_resampler is None:
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=frame.num_channels,
            )
        return self._input_resampler.push(frame)

    def _audio_owner(self) -> _Reply | None:
        """The reply Fish Audio's output belongs to right now, if any."""
        pending: _Reply | None = None
        speaking: _Reply | None = None
        for reply in self._replies.values():
            if not reply.exposed or reply.done or reply.ambient:
                continue
            if reply.started:
                speaking = reply
            elif reply.sent_first and not reply.closed:
                pending = reply
        return speaking or pending

    def _on_audio(self, data: bytes) -> None:
        # between replies Fish Audio sends exact zeros, so any other sample is sound
        silent = data.count(0) == len(data)
        duration_ms = len(data) / 2 / SAMPLE_RATE * 1000

        owner = self._audio_owner()
        if owner is not None:
            self._close_ambient()
            if owner.closed:
                # the framework stopped this reply; what is left of its sound is not played
                return
            if silent:
                # a pause is played only if speech follows it, so a reply ends with its speech
                if owner.has_audio:
                    owner.held_silence.append(data)
                return
            for held in owner.held_silence:
                self._forward_audio(owner, held)
            owner.held_silence = []
            self._forward_audio(owner, data)
            return

        ambient = self._ambient
        if self._ambient_muted:
            # a generation plays one after another, and the tool call is behind this one
            if ambient is not None:
                if not silent:
                    self._forward_audio(ambient, _fade_out(data))
                self._close_ambient()
            return

        if silent:
            if ambient is None:
                return
            self._ambient_silence_ms += duration_ms
            if self._ambient_silence_ms >= _AMBIENT_SILENCE_MS:
                self._close_ambient()
                return
            self._forward_audio(ambient, data)
            return

        # the voice's own sound outside a reply: a backchannel, a breath
        if ambient is None:
            ambient = self._new_reply(user_initiated=False)
            ambient.ambient = True
            self._ambient = ambient
            self._expose(ambient, modalities=["audio"])
        self._ambient_silence_ms = 0.0
        self._forward_audio(ambient, data)

    def _forward_audio(self, reply: _Reply, data: bytes) -> None:
        if reply.audio_ch.closed:
            return

        now = time.time()
        if now - self._last_forward_at > _PLAYOUT_IDLE:
            cushion_samples = SAMPLE_RATE * _JITTER_BUFFER_MS // 1000
            reply.audio_ch.send_nowait(_audio_frame(bytes(cushion_samples * 2)))
        self._last_forward_at = now

        if not reply.has_audio:
            reply.has_audio = True
            reply.first_audio_at = time.time()
        reply.audio_ch.send_nowait(_audio_frame(data))

    def _close_ambient(self) -> None:
        ambient = self._ambient
        if ambient is None:
            return
        self._ambient = None
        self._ambient_silence_ms = 0.0
        ambient.done = True
        self._close_generation(ambient)
        self._replies.pop(ambient.id, None)

    # -- server events -----------------------------------------------------------------

    def _handle_event(self, event: dict[str, Any]) -> None:
        self.emit("fishaudio_server_event_received", event)
        event_type = event.get("type")

        if event_type == "session.started":
            self._on_session_started(event)
        elif event_type == "turn.speech_started":
            self._turn_started_at[str(event.get("turn_id"))] = time.time()
            self._caller_heard_at = time.time()
            self._ambient_muted = False
            self.emit("input_speech_started", llm.InputSpeechStartedEvent())
        elif event_type == "turn.speech_ended":
            self.emit(
                "input_speech_stopped",
                llm.InputSpeechStoppedEvent(user_transcription_enabled=True),
            )
        elif event_type == "turn.interim":
            self._on_turn_interim(event)
        elif event_type == "turn.eager":
            self._on_turn_eager(event)
        elif event_type == "turn.resumed":
            self._on_turn_resumed(event)
        elif event_type == "turn.committed":
            self._on_turn_committed(event)
        elif event_type == "turn.dropped":
            self._on_turn_dropped(event)
        elif event_type == "turn.backchannel":
            self._last_backchannel = str(event.get("text") or "").strip()
        elif event_type == "reply.started":
            self._on_reply_started(event)
        elif event_type == "reply.progress":
            self._on_reply_progress(event)
        elif event_type == "reply.interrupted":
            self._on_reply_interrupted(event)
        elif event_type == "reply.done":
            self._on_reply_done(event)
        elif event_type == "error":
            self._on_error(event)

    def _on_session_started(self, event: dict[str, Any]) -> None:
        self._session_id = event.get("session_id")
        self._text_rules = event.get("text_rules") or {}
        limits = event.get("limits") or {}
        self._max_text_bytes = int(limits.get("max_text_bytes") or _DEFAULT_MAX_TEXT_BYTES)
        self._connected = True
        self._was_connected = True

        pending = self._pending_frames
        self._pending_frames = []
        for frame in pending:
            self._send(frame)

    def _on_error(self, event: dict[str, Any]) -> None:
        self._last_error = f"{event.get('code')}: {event.get('message')}"
        if event.get("recoverable"):
            logger.warning("Fish Audio reported an error", extra={"error": self._last_error})
            return
        # the server closes the session next, and the close is what gets reported

    def _item_id(self, turn_id: str) -> str:
        return f"fish_{self._conn_tag}_{turn_id}"

    def _on_turn_interim(self, event: dict[str, Any]) -> None:
        self._caller_heard_at = time.time()
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=self._item_id(str(event.get("turn_id"))),
                transcript=str(event.get("text") or ""),
                is_final=False,
            ),
        )

    def _on_turn_eager(self, event: dict[str, Any]) -> None:
        eager_id = str(event.get("eager_id") or "")
        text = str(event.get("text") or "").strip()
        if not eager_id or not text:
            return
        self._caller_heard_at = time.time()

        self._cancel_speculative()
        reply = self._new_reply(user_initiated=False)
        reply.eager_id = eager_id
        self._speculative = reply

        user_texts: list[str] = []
        for held_text, _ in self._held_asides:
            user_texts.append(held_text)
        user_texts.append(text)
        self._start_llm(reply, user_texts=user_texts)

    def _on_turn_resumed(self, event: dict[str, Any]) -> None:
        self._caller_heard_at = time.time()
        speculative = self._speculative
        if speculative is not None and speculative.eager_id == event.get("eager_id"):
            self._cancel_speculative()

    def _on_turn_committed(self, event: dict[str, Any]) -> None:
        turn_id = str(event.get("turn_id") or "")
        text = str(event.get("text") or "").strip()
        self._caller_heard_at = None
        if not text:
            self._cancel_speculative()
            return

        if event.get("kind") == "aside":
            self._on_aside(text)
            return

        # a real turn: the reply it cut is not picked up again
        self._cut = None
        self._take_held_asides()
        self._add_user_turn(self._item_id(turn_id), text, self._turn_started_at.pop(turn_id, None))

        speculative = self._speculative
        approved = event.get("approved_eager_id")
        if speculative is not None and approved and speculative.eager_id == approved:
            self._speculative = None
            self._expose(speculative)
            return

        self._cancel_speculative()
        reply = self._new_reply(user_initiated=False)
        self._expose(reply)
        self._start_llm(reply)

    def _on_aside(self, text: str) -> None:
        # speech not meant for the agent stops nothing and gets no reply of its own
        over_reply = self._reply_in_flight()
        self._cancel_speculative()

        # Fish Audio already called an "mm-hmm" a backchannel; it asks for nothing
        backchannel = text == self._last_backchannel
        self._last_backchannel = ""

        if (over_reply or self._held_asides) and _HAS_WORDS.search(text) and not backchannel:
            # said over a reply, it may be the user talking over the agent without Fish
            # Audio taking it for a turn: the LLM gets the words once the reply has ended
            self._held_asides.append((text, time.time()))
            if self._aside_task is None or self._aside_task.done():
                self._aside_task = asyncio.create_task(self._release_asides_task())
            return

        self._resume_cut()

    def _on_turn_dropped(self, event: dict[str, Any]) -> None:
        self._caller_heard_at = None
        if self._cut is None:
            return
        # the sound that stopped the reply was not a turn
        self._cancel_speculative()
        self._resume_cut()

    def _on_reply_started(self, event: dict[str, Any]) -> None:
        reply = self._replies.get(str(event.get("reply_id")))
        if reply is None:
            return
        reply.started = True
        self._ambient_muted = False

    def _on_reply_progress(self, event: dict[str, Any]) -> None:
        reply = self._replies.get(str(event.get("reply_id")))
        if reply is None:
            return
        spoken_chars = int(event.get("spoken_chars") or 0)
        self._push_heard_text(reply, reply.sent_text[:spoken_chars])

    def _on_reply_interrupted(self, event: dict[str, Any]) -> None:
        reply = self._replies.get(str(event.get("reply_id")))
        if reply is None:
            return
        reply.wire_stopped = True
        if event.get("by") == "caller":
            self._caller_heard_at = time.time()
        self._push_heard_text(reply, str(event.get("heard_text") or ""))

    def _on_reply_done(self, event: dict[str, Any]) -> None:
        reply = self._replies.pop(str(event.get("reply_id")), None)
        if reply is None:
            return
        reply.done = True
        reply.wire_stopped = True
        if reply is self._speculative:
            self._speculative = None

        status = event.get("status")
        heard_text = str(event.get("heard_text") or "")
        unheard_text = str(event.get("unheard_text") or "")

        if heard_text.strip():
            self._push_heard_text(reply, heard_text)
            message = llm.ChatMessage(
                id=reply.message_id,
                role="assistant",
                content=[heard_text],
                created_at=reply.first_audio_at or time.time(),
                interrupted=status == "interrupted",
            )
            self._insert_item(message)

        if status == "failed":
            logger.warning(
                "Fish Audio could not speak a reply",
                extra={"reason": event.get("reason"), "reply_id": reply.id},
            )

        cut_by_caller = status == "interrupted" and event.get("by") == "caller"
        stopped_early = status == "partial"
        if (cut_by_caller or stopped_early) and reply.exposed and not reply.cancelled:
            self._cut = _Cut(reply=reply, unheard_text=unheard_text, created_at=time.time())
            if stopped_early:
                # Fish Audio stopped by itself, so nothing is coming that would resume it
                if reply.resumed:
                    self._cut = None
                else:
                    self._resume_cut()

        self._close_generation(reply)

    # -- replies -----------------------------------------------------------------------

    def _new_reply(self, *, user_initiated: bool) -> _Reply:
        reply = _Reply(
            id=utils.shortuuid("reply_"),
            message_id=utils.shortuuid("item_"),
            user_initiated=user_initiated,
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            text_ch=utils.aio.Chan(),
            audio_ch=utils.aio.Chan(),
        )
        self._replies[reply.id] = reply
        return reply

    def _expose(
        self, reply: _Reply, *, modalities: list[Literal["text", "audio"]] | None = None
    ) -> None:
        """Hand the reply to the framework as a generation."""
        reply.exposed = True
        reply.eager_id = None

        event = llm.GenerationCreatedEvent(
            message_stream=reply.message_ch,
            function_stream=reply.function_ch,
            user_initiated=reply.user_initiated,
            response_id=reply.id,
        )
        if reply.fut is not None and not reply.fut.done():
            reply.fut.set_result(event)
        self.emit("generation_created", event)

        if reply.sent_first or modalities is not None:
            self._send_message(reply, modalities=modalities)

        held_calls = reply.held_calls
        reply.held_calls = []
        for call in held_calls:
            self._deliver_call(reply, call)

        self._maybe_close(reply)

    def _send_message(
        self, reply: _Reply, *, modalities: list[Literal["text", "audio"]] | None = None
    ) -> None:
        if reply.message_sent or reply.message_ch.closed:
            return
        reply.message_sent = True

        modalities_fut: asyncio.Future[list[Literal["text", "audio"]]] = asyncio.Future()
        modalities_fut.set_result(modalities or ["audio", "text"])
        reply.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=reply.message_id,
                text_stream=reply.text_ch,
                audio_stream=reply.audio_ch,
                modalities=modalities_fut,
            )
        )

    def _push_heard_text(self, reply: _Reply, heard_text: str) -> None:
        if reply.text_ch.closed:
            return
        delta = heard_text[reply.pushed_chars :]
        if not delta:
            return
        reply.pushed_chars = len(heard_text)
        reply.text_ch.send_nowait(delta)

    def _deliver_call(self, reply: _Reply, call: llm.FunctionCall) -> None:
        if reply.function_ch.closed:
            logger.warning(
                "dropping a tool call from a reply that already ended",
                extra={"function": call.name, "reply_id": reply.id},
            )
            return
        # a call written for an eager turn is placed after the turn that committed it
        call.created_at = time.time()
        self._insert_item(call)
        reply.function_ch.send_nowait(call)
        if not reply.sent_first:
            self._ambient_muted = True

    def _maybe_close(self, reply: _Reply) -> None:
        """Close the generation once nothing more can come of it."""
        nothing_to_say = reply.llm_done and not reply.sent_first
        if reply.exposed and (reply.done or nothing_to_say):
            reply.done = True
            self._close_generation(reply)
            self._replies.pop(reply.id, None)

    def _close_generation(self, reply: _Reply) -> None:
        if reply.closed:
            return
        reply.closed = True
        reply.text_ch.close()
        reply.audio_ch.close()
        reply.message_ch.close()
        reply.function_ch.close()
        if reply.fut is not None and not reply.fut.done():
            reply.fut.set_exception(llm.RealtimeError("the reply ended before it started"))
        if reply.exposed and not reply.ambient:
            self._emit_metrics(reply)

    def _cancel_reply(self, reply: _Reply) -> None:
        reply.cancelled = True
        if reply.llm_task is not None:
            reply.llm_task.cancel()
        if reply.sent_first and not reply.done:
            self._send({"type": "reply.cancel", "reply_id": reply.id})
        reply.wire_stopped = True
        self._close_generation(reply)
        if not reply.sent_first:
            # nothing reached Fish Audio, so no reply.done will come for it
            self._replies.pop(reply.id, None)

    def _cancel_speculative(self) -> None:
        speculative = self._speculative
        if speculative is None:
            return
        self._speculative = None
        self._cancel_reply(speculative)

    def _reply_in_flight(self) -> bool:
        """Whether a reply is being said, or written for a turn Fish Audio committed."""
        for reply in self._replies.values():
            if reply.exposed and not reply.done and not reply.ambient:
                return True
        return False

    def _reset_replies(self, reason: str) -> None:
        for reply in list(self._replies.values()):
            reply.cancelled = True
            if reply.llm_task is not None:
                reply.llm_task.cancel()
            if reply.fut is not None and not reply.fut.done():
                reply.fut.set_exception(llm.RealtimeError(reason))
            self._close_generation(reply)
        self._replies.clear()
        self._speculative = None
        self._ambient = None
        self._ambient_muted = False
        self._cut = None
        self._held_asides = []
        self._last_backchannel = ""
        self._pending_frames = []
        self._turn_started_at = {}
        self._caller_heard_at = None

    # -- writing replies ---------------------------------------------------------------

    def _create_task(self, coro: Any) -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _start_llm(
        self,
        reply: _Reply,
        *,
        user_texts: list[str] | None = None,
        instructions: str | None = None,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> None:
        chat_ctx = self._build_llm_chat_ctx(user_texts=user_texts, instructions=instructions)
        if not is_given(tool_choice) and self._tool_choice is not None:
            tool_choice = self._tool_choice
        tool_list = tools if is_given(tools) else self._tools.flatten()
        reply.llm_task = self._create_task(
            self._llm_task(reply, chat_ctx=chat_ctx, tools=tool_list, tool_choice=tool_choice)
        )

    def _build_llm_chat_ctx(
        self, *, user_texts: list[str] | None, instructions: str | None
    ) -> llm.ChatContext:
        chat_ctx = self._chat_ctx.copy()

        system_parts: list[str] = []
        if self._instructions:
            system_parts.append(self._instructions)
        if self._opts.text_rules:
            system_parts.append(_text_rules_prompt(self._text_rules))
        if system_parts:
            system_message = llm.ChatMessage(
                role="system", content=["\n\n".join(system_parts)], created_at=0.0
            )
            chat_ctx.items.insert(0, system_message)

        for user_text in user_texts or []:
            chat_ctx.add_message(role="user", content=user_text)
        if instructions:
            chat_ctx.add_message(role="system", content=instructions)
        return chat_ctx

    @utils.log_exceptions(logger=logger)
    async def _llm_task(
        self,
        reply: _Reply,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        tool_choice: NotGivenOr[llm.ToolChoice],
    ) -> None:
        try:
            stream = self._llm.chat(
                chat_ctx=chat_ctx,
                tools=tools,
                tool_choice=tool_choice,
                conn_options=self._opts.conn_options,
            )
            async with stream:
                async for chunk in stream:
                    if chunk.usage is not None:
                        reply.usage = chunk.usage
                    delta = chunk.delta
                    if delta is None:
                        continue
                    if delta.content:
                        self._send_reply_text(reply, delta.content)
                    for tool_call in delta.tool_calls:
                        self._on_tool_call(reply, tool_call)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # one reply failed; the session itself still works
            self._emit_error(e, recoverable=True)

        reply.llm_done = True
        self._end_reply_text(reply)
        self._maybe_close(reply)

    async def _say_task(self, reply: _Reply, text: str | AsyncIterable[str]) -> None:
        if isinstance(text, str):
            self._send_reply_text(reply, text)
        else:
            async for chunk in text:
                self._send_reply_text(reply, chunk)

        reply.llm_done = True
        self._end_reply_text(reply)
        self._maybe_close(reply)

    def _on_tool_call(self, reply: _Reply, tool_call: llm.FunctionToolCall) -> None:
        call = llm.FunctionCall(
            call_id=tool_call.call_id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            extra=tool_call.extra or {},
        )
        if reply.sent_first and not reply.wire_stopped:
            # say what was written before the call while the tool runs
            self._send({"type": "reply.text", "reply_id": reply.id, "text": "", "flush": True})

        if reply.exposed:
            self._deliver_call(reply, call)
        else:
            reply.held_calls.append(call)

    def _send_reply_text(self, reply: _Reply, text: str) -> None:
        if not text:
            return
        if reply.wire_stopped:
            reply.unsent_tail += text
            return

        for piece in _split_utf8(text, self._max_text_bytes):
            frame: dict[str, Any] = {"type": "reply.text", "reply_id": reply.id, "text": piece}
            if not reply.sent_first:
                reply.sent_first = True
                if reply.eager_id is not None:
                    # Fish Audio holds the speech until it commits the turn
                    frame["for"] = reply.eager_id
            self._send(frame)

        reply.sent_text += text
        if reply.exposed:
            self._send_message(reply)

    def _end_reply_text(self, reply: _Reply) -> None:
        if not reply.sent_first or reply.sent_last or reply.wire_stopped:
            return
        reply.sent_last = True
        self._send({"type": "reply.text", "reply_id": reply.id, "text": "", "last": True})

    # -- turns the caller did not take ------------------------------------------------

    def _add_user_turn(self, item_id: str, text: str, started_at: float | None) -> None:
        message = llm.ChatMessage(
            id=item_id,
            role="user",
            content=[text],
            created_at=started_at or time.time(),
        )
        self._insert_item(message)
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=item_id,
                transcript=text,
                is_final=True,
                turn_started_at=started_at,
            ),
        )

    def _insert_item(self, item: llm.ChatItem) -> None:
        self._unacked_item_ids.add(item.id)
        self._chat_ctx.insert(item)

    def _resume_cut(self) -> None:
        cut = self._cut
        self._cut = None
        if cut is None or self._closing:
            return
        if time.time() - cut.created_at > _CUT_RESUME_WINDOW:
            return
        self._create_task(self._resume_task(cut))

    async def _resume_task(self, cut: _Cut) -> None:
        llm_task = cut.reply.llm_task
        if llm_task is not None and not llm_task.done():
            # the LLM kept writing after the cut: what it adds belongs to the same reply
            await asyncio.wait([llm_task])

        text = cut.unheard_text + cut.reply.unsent_tail
        if not text.strip() or self._closing:
            return

        reply = self._new_reply(user_initiated=False)
        reply.resumed = True
        self._expose(reply)
        self._send_reply_text(reply, text)
        reply.llm_done = True
        self._end_reply_text(reply)

    def _take_held_asides(self) -> None:
        held = self._held_asides
        self._held_asides = []
        for text, heard_at in held:
            self._add_user_turn(utils.shortuuid("fish_aside_"), text, heard_at)

    async def _release_asides_task(self) -> None:
        while self._held_asides:
            await asyncio.sleep(_ASIDE_POLL_INTERVAL)
            caller_heard_at = self._caller_heard_at
            caller_may_commit = (
                caller_heard_at is not None and time.time() - caller_heard_at < _CALLER_COMMIT_WAIT
            )
            if caller_may_commit or self._reply_in_flight():
                continue
            if not self._held_asides:
                return

            # nothing is in the way any more: the words get an answer of their own
            self._cut = None
            self._take_held_asides()
            reply = self._new_reply(user_initiated=False)
            self._expose(reply)
            self._start_llm(reply)

    # -- metrics -----------------------------------------------------------------------

    def _emit_metrics(self, reply: _Reply) -> None:
        duration = time.time() - reply.created_at
        ttft = -1.0
        if reply.first_audio_at is not None:
            ttft = reply.first_audio_at - reply.created_at

        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        if reply.usage is not None:
            input_tokens = reply.usage.prompt_tokens
            output_tokens = reply.usage.completion_tokens
            cached_tokens = reply.usage.prompt_cached_tokens

        tokens_per_second = 0.0
        if duration > 0:
            tokens_per_second = output_tokens / duration

        self.emit(
            "metrics_collected",
            RealtimeModelMetrics(
                request_id=reply.id,
                timestamp=reply.created_at,
                label=self._realtime_model.label,
                duration=duration,
                ttft=ttft,
                cancelled=reply.cancelled,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                tokens_per_second=tokens_per_second,
                input_token_details=RealtimeModelMetrics.InputTokenDetails(
                    text_tokens=input_tokens,
                    cached_tokens=cached_tokens,
                ),
                output_token_details=RealtimeModelMetrics.OutputTokenDetails(
                    text_tokens=output_tokens,
                ),
                metadata=Metadata(
                    model_name=self._llm.model,
                    model_provider=self._llm.provider,
                ),
            ),
        )


def _audio_frame(data: bytes) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=data,
        sample_rate=SAMPLE_RATE,
        num_channels=NUM_CHANNELS,
        samples_per_channel=len(data) // 2,
    )


def _fade_out(data: bytes) -> bytes:
    """The start of ``data``, faded to silence."""
    samples = np.frombuffer(data, dtype=np.int16)
    fade_samples = min(len(samples), SAMPLE_RATE * _FADE_OUT_MS // 1000)
    ramp = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    faded = samples[:fade_samples].astype(np.float32) * ramp
    return faded.astype(np.int16).tobytes()


def _split_utf8(text: str, max_bytes: int) -> list[str]:
    """Split text into pieces of at most ``max_bytes`` UTF-8 bytes each."""
    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    pieces: list[str] = []
    piece = ""
    piece_bytes = 0
    for char in text:
        char_bytes = len(char.encode("utf-8"))
        if piece_bytes + char_bytes > max_bytes:
            pieces.append(piece)
            piece = ""
            piece_bytes = 0
        piece += char
        piece_bytes += char_bytes
    if piece:
        pieces.append(piece)
    return pieces


def _text_rules_prompt(text_rules: dict[str, Any]) -> str:
    """Fish Audio's rules for text its voice will speak, as lines for the system prompt."""
    lines = [
        "Your replies are spoken aloud word for word by a voice model.",
        "- Write only the words to say. No markdown, lists, emojis or stage directions.",
        "- Chat history shows only what the user heard of your replies before interrupting.",
    ]

    if text_rules.get("numbers") == "spoken":
        lines.append("- Write numbers as spoken words, for example 'twenty five'.")
    if text_rules.get("ack_in_t") is False:
        lines.append(
            "- Do not open a reply with an acknowledgement such as 'Okay,', 'Sure,' or "
            "'Got it,'. The voice adds its own."
        )

    avoid = text_rules.get("avoid") or []
    if "colon" in avoid:
        lines.append("- Do not put a colon after a word. Use a comma instead.")
    if "semicolon" in avoid:
        lines.append("- Do not use semicolons. Use commas or separate sentences.")
    if "spaced_dash" in avoid:
        lines.append("- Do not use dashes between words. Use commas instead.")
    return "\n".join(lines)
