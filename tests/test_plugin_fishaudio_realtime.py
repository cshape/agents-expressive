from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, llm, utils
from livekit.agents.utils.aio.channel import ChanEmpty
from livekit.plugins.fishaudio.realtime import RealtimeModel, RealtimeSession
from livekit.plugins.fishaudio.realtime.realtime_model import (
    SAMPLE_RATE,
    _split_utf8,
    _text_rules_prompt,
)

from .fake_llm import FakeLLM, FakeLLMResponse

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time]

_QUESTION = "What time do you open?"
_ANSWER = "We open at nine. Anything else?"
_HEARD = "We open at nine."
_UNHEARD = " Anything else?"


class _Harness:
    """A session with no connection: events go in by hand, frames come out of a channel."""

    def __init__(self, session: RealtimeSession) -> None:
        self.session = session
        self.sent: utils.aio.Chan[dict[str, Any]] = utils.aio.Chan()
        self.generations: list[llm.GenerationCreatedEvent] = []
        self.transcripts: list[llm.InputTranscriptionCompleted] = []
        session._connected = True
        session._send_ch = self.sent
        session.on("generation_created", self.generations.append)
        session.on("input_audio_transcription_completed", self.transcripts.append)

    def frames(self) -> list[dict[str, Any]]:
        frames = []
        while True:
            try:
                frames.append(self.sent.recv_nowait())
            except ChanEmpty:
                return frames

    def event(self, event_type: str, **fields: Any) -> None:
        event = {"type": event_type, "seq": 0}
        event.update(fields)
        self.session._handle_event(event)

    def eager(self, text: str = _QUESTION, eager_id: str = "e1", turn_id: str = "t0") -> None:
        self.event("turn.eager", eager_id=eager_id, turn_id=turn_id, text=text)

    def committed(
        self,
        text: str = _QUESTION,
        *,
        approved_eager_id: str | None = "e1",
        turn_id: str = "t0",
        kind: str = "turn",
    ) -> None:
        self.event(
            "turn.committed",
            turn_id=turn_id,
            text=text,
            kind=kind,
            approved_eager_id=approved_eager_id,
        )

    def done(self, reply_id: str, heard: str, unheard: str = "", **fields: Any) -> None:
        payload = {"status": "completed", "by": None, "reason": None}
        payload.update(fields)
        self.event(
            "reply.done", reply_id=reply_id, heard_text=heard, unheard_text=unheard, **payload
        )


def _response(input: str, content: str, **kwargs: Any) -> FakeLLMResponse:
    return FakeLLMResponse(input=input, content=content, ttft=0.0, duration=0.0, **kwargs)


@pytest.fixture
def make_harness(monkeypatch: pytest.MonkeyPatch):
    async def _no_connection(self: RealtimeSession) -> None:
        return None

    monkeypatch.setattr(RealtimeSession, "_main_task", _no_connection)
    sessions: list[RealtimeSession] = []

    def _make(*responses: FakeLLMResponse, **model_kwargs: Any) -> _Harness:
        fake_llm = FakeLLM(fake_responses=list(responses))
        model = RealtimeModel(
            llm=fake_llm, api_key="test-key", http_session=MagicMock(), **model_kwargs
        )
        session = model.session()
        sessions.append(session)
        return _Harness(session)

    return _make


async def _settle() -> None:
    await asyncio.sleep(0.05)


def _reply_text(frames: list[dict[str, Any]]) -> str:
    text = ""
    for frame in frames:
        if frame["type"] == "reply.text":
            text += frame["text"]
    return text


async def _collect(stream: Any) -> list[Any]:
    items = []
    async for item in stream:
        items.append(item)
    return items


def _silence(ms: int) -> bytes:
    return bytes(SAMPLE_RATE * 2 * ms // 1000)


def _sound(ms: int) -> bytes:
    return b"\x10\x20" * (SAMPLE_RATE * ms // 1000)


# ---------------------------------------------------------------------------
# turns
# ---------------------------------------------------------------------------


async def test_eager_turn_writes_a_held_reply(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()

    frames = harness.frames()
    assert _reply_text(frames) == _ANSWER
    # only the first frame names what the reply answers
    assert frames[0]["for"] == "e1"
    for frame in frames[1:]:
        assert "for" not in frame
    assert frames[-1]["last"] is True
    # Fish Audio has not committed the turn, so the framework knows of no reply
    assert harness.generations == []
    assert harness.transcripts == []


async def test_committed_turn_hands_the_reply_to_the_framework(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    harness.committed()

    assert len(harness.generations) == 1
    assert harness.generations[0].user_initiated is False
    assert harness.transcripts[-1].is_final is True
    assert harness.transcripts[-1].transcript == _QUESTION

    chat_ctx = harness.session.chat_ctx
    assert chat_ctx.items[-1].role == "user"
    assert chat_ctx.items[-1].text_content == _QUESTION
    # the user message keeps the id the framework was given for it
    assert chat_ctx.items[-1].id == harness.transcripts[-1].item_id


async def test_resumed_turn_cancels_the_held_reply(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]

    harness.event("turn.resumed", eager_id="e1")

    assert harness.frames() == [{"type": "reply.cancel", "reply_id": reply_id}]
    assert harness.generations == []


async def test_turn_committed_without_its_eager_starts_a_new_reply(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager(text="What time")
    await _settle()
    harness.frames()

    harness.committed(approved_eager_id=None)
    await _settle()

    frames = harness.frames()
    assert _reply_text(frames) == _ANSWER
    # the turn is committed, so nothing is held
    assert "for" not in frames[0]
    assert len(harness.generations) == 1


# ---------------------------------------------------------------------------
# what the user heard
# ---------------------------------------------------------------------------


async def _first_message(generation: llm.GenerationCreatedEvent) -> llm.MessageGeneration:
    async for message in generation.message_stream:
        return message
    raise AssertionError("the generation has no message")


async def test_interrupted_reply_keeps_what_was_heard(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    message = await _first_message(harness.generations[0])

    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    harness.event("reply.progress", reply_id=reply_id, spoken_chars=7)
    harness.event(
        "reply.interrupted", reply_id=reply_id, by="caller", heard_text=_HEARD, final=True
    )
    harness.done(reply_id, _HEARD, _UNHEARD, status="interrupted", by="caller")

    text = "".join(await _collect(message.text_stream))
    assert text == _HEARD

    assistant = harness.session.chat_ctx.items[-1]
    assert assistant.role == "assistant"
    assert assistant.text_content == _HEARD
    assert assistant.interrupted is True
    # the framework records the message under the same id
    assert assistant.id == message.message_id


async def test_reply_cut_by_noise_is_picked_up_again(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    harness.done(reply_id, _HEARD, _UNHEARD, status="interrupted", by="caller")

    # the sound that stopped the reply was not a turn
    harness.event("turn.dropped", reason="nonuser")
    await _settle()

    frames = harness.frames()
    assert _reply_text(frames) == _UNHEARD
    assert frames[0]["reply_id"] != reply_id
    assert len(harness.generations) == 2


async def test_reply_cut_by_a_real_turn_is_not_picked_up_again(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER), _response("Stop.", "Sure."))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    harness.done(reply_id, _HEARD, _UNHEARD, status="interrupted", by="caller")

    harness.committed("Stop.", approved_eager_id=None, turn_id="t1")
    await _settle()

    assert _reply_text(harness.frames()) == "Sure."


async def test_cut_reply_does_not_resume_over_a_real_turn(make_harness):
    harness = make_harness(
        # still writing when the caller cuts it
        FakeLLMResponse(input=_QUESTION, content=_ANSWER, ttft=0.0, duration=2.0),
        _response("Stop.", "Sure."),
    )

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    harness.done(reply_id, _HEARD, _UNHEARD, status="interrupted", by="caller")
    harness.event("turn.dropped", reason="nonuser")

    # before the old reply finished writing, the caller takes a real turn
    harness.committed("Stop.", approved_eager_id=None, turn_id="t1")
    await asyncio.sleep(5)

    assert _reply_text(harness.frames()) == "Sure."


async def test_new_turn_cancels_a_reply_not_yet_spoken(make_harness):
    harness = make_harness(
        FakeLLMResponse(input=_QUESTION, content=_ANSWER, ttft=2.0, duration=2.0),
        _response("And on Sunday?", "Ten on Sunday."),
    )

    harness.committed(approved_eager_id=None)
    await _settle()
    # the caller adds to the question before the first answer is written
    harness.committed("And on Sunday?", approved_eager_id=None, turn_id="t1")
    await asyncio.sleep(5)

    assert _reply_text(harness.frames()) == "Ten on Sunday."


async def test_dropped_turn_cancels_its_held_reply(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]

    harness.event("turn.dropped", reason="nonuser")

    assert harness.frames() == [{"type": "reply.cancel", "reply_id": reply_id}]


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------


async def test_tool_call_written_after_a_cut_does_not_run(make_harness):
    tool_call = llm.FunctionToolCall(name="book", arguments="{}", call_id="c1")
    harness = make_harness(
        FakeLLMResponse(
            input=_QUESTION, content=_ANSWER, ttft=0.0, duration=2.0, tool_calls=[tool_call]
        ),
        _response("Stop.", "Sure."),
    )

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    # the caller cuts the reply before the LLM gets to its tool call
    harness.event(
        "reply.interrupted", reply_id=reply_id, by="caller", heard_text=_HEARD, final=True
    )
    harness.done(reply_id, _HEARD, _UNHEARD, status="interrupted", by="caller")
    harness.committed("Stop.", approved_eager_id=None, turn_id="t1")
    await asyncio.sleep(5)

    for item in harness.session.chat_ctx.items:
        assert item.type != "function_call"


async def test_tool_call_waits_for_the_turn_to_be_committed(make_harness):
    tool_call = llm.FunctionToolCall(name="add_item", arguments='{"item": "tea"}', call_id="c1")
    harness = make_harness(_response(_QUESTION, "", tool_calls=[tool_call]))

    harness.eager()
    await _settle()
    assert harness.generations == []

    harness.committed()

    calls = await _collect(harness.generations[0].function_stream)
    assert len(calls) == 1
    assert calls[0].name == "add_item"
    assert calls[0].call_id == "c1"
    # the call is in the chat context, where the framework adds its output
    assert harness.session.chat_ctx.items[-1].type == "function_call"


async def test_tool_call_of_a_resumed_turn_never_runs(make_harness):
    tool_call = llm.FunctionToolCall(name="add_item", arguments="{}", call_id="c1")
    harness = make_harness(_response(_QUESTION, "", tool_calls=[tool_call]))

    harness.eager()
    await _settle()
    harness.event("turn.resumed", eager_id="e1")

    assert harness.generations == []
    assert harness.session.chat_ctx.items == []


async def test_generate_reply_answers_a_tool_result(make_harness):
    harness = make_harness(_response("12 degrees", "It is twelve degrees."))

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="user", content="Weather?")
    chat_ctx.items.append(llm.FunctionCall(call_id="c1", name="weather", arguments="{}"))
    chat_ctx.items.append(
        llm.FunctionCallOutput(call_id="c1", name="weather", output="12 degrees", is_error=False)
    )
    await harness.session.update_chat_ctx(chat_ctx)

    generation = await harness.session.generate_reply()
    await _settle()

    assert generation.user_initiated is True
    frames = harness.frames()
    assert _reply_text(frames) == "It is twelve degrees."
    assert "for" not in frames[0]


# ---------------------------------------------------------------------------
# speech the user did not address to the agent
# ---------------------------------------------------------------------------


async def test_aside_over_a_reply_is_answered_after_it(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER), _response("Is it far?", "Not far."))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    harness.session._on_audio(_sound(40))
    await _settle()

    harness.committed("Is it far?", approved_eager_id=None, turn_id="t1", kind="aside")
    await _settle()
    # an aside stops nothing and is not answered while the reply is being said
    assert harness.frames() == []

    harness.done(reply_id, _ANSWER)
    await asyncio.sleep(0.5)

    assert _reply_text(harness.frames()) == "Not far."
    assert harness.transcripts[-1].transcript == "Is it far?"


async def test_backchannel_over_a_reply_is_ignored(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)

    harness.event("turn.backchannel", text="Mm-hmm.", at_ms=2000)
    harness.committed("Mm-hmm.", approved_eager_id=None, turn_id="t1", kind="aside")
    harness.done(reply_id, _ANSWER)
    await asyncio.sleep(0.5)

    assert harness.frames() == []
    assert len(harness.generations) == 1


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


async def test_reply_audio_ends_with_its_speech(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    message = await _first_message(harness.generations[0])

    harness.session._on_audio(_silence(40))
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)
    harness.session._on_audio(_sound(40))
    harness.session._on_audio(_silence(40))
    harness.session._on_audio(_sound(40))
    for _ in range(10):
        harness.session._on_audio(_silence(100))
    harness.done(reply_id, _ANSWER)

    frames: list[rtc.AudioFrame] = await _collect(message.audio_stream)
    durations = [round(frame.duration * 1000) for frame in frames]
    # a cushion of silence, the speech with its pause, then only a little of the silence
    # after it: the rest would keep the playout busy for nothing
    assert durations == [200, 40, 40, 40, 100, 100, 100]


async def test_voice_sound_between_replies_is_played(make_harness):
    harness = make_harness()

    harness.session._on_audio(_silence(100))
    assert harness.generations == []

    harness.session._on_audio(_sound(40))
    assert len(harness.generations) == 1
    message = await _first_message(harness.generations[0])
    assert await message.modalities == ["audio"]

    for _ in range(10):
        harness.session._on_audio(_silence(40))

    frames = await _collect(message.audio_stream)
    assert frames
    assert await _collect(message.text_stream) == []


async def test_interrupt_cancels_the_reply_and_drops_its_audio(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.event("reply.started", reply_id=reply_id, at_ms=1000)

    harness.session.interrupt()
    assert harness.frames() == [{"type": "reply.cancel", "reply_id": reply_id}]

    # what is left of the reply's sound does not come back as a new generation
    harness.session._on_audio(_sound(40))
    assert len(harness.generations) == 1

    harness.done(reply_id, "We open", " at nine.", status="interrupted", by="client")
    assert harness.session.chat_ctx.items[-1].text_content == "We open"


async def test_push_audio_resamples_to_the_session_rate(make_harness):
    harness = make_harness()

    frame = rtc.AudioFrame(
        data=bytes(48000 * 2 // 10), sample_rate=48000, num_channels=1, samples_per_channel=4800
    )
    for _ in range(2):
        harness.session.push_audio(frame)

    buffered_ms = len(harness.session._input_buf) / 2 / SAMPLE_RATE * 1000
    assert 150 <= buffered_ms <= 200


async def test_caller_audio_is_never_held_for_long(make_harness):
    harness = make_harness()

    frame = rtc.AudioFrame(
        data=bytes(SAMPLE_RATE * 2 // 10),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLE_RATE // 10,
    )
    for _ in range(20):
        harness.session.push_audio(frame)

    # old caller audio is dropped: Fish Audio would hear the caller that much late
    buffered_ms = len(harness.session._input_buf) / 2 / SAMPLE_RATE * 1000
    assert buffered_ms == 300

    harness.event("session.started", session_id="s1", limits={}, text_rules={})
    assert len(harness.session._input_buf) == 0


# ---------------------------------------------------------------------------
# say, context, options
# ---------------------------------------------------------------------------


async def test_say_speaks_the_text_as_given(make_harness):
    harness = make_harness()

    generation = await harness.session.say("Please hold.")
    await _settle()

    assert generation.user_initiated is True
    frames = harness.frames()
    assert _reply_text(frames) == "Please hold."
    assert frames[-1]["last"] is True


async def test_say_with_a_failing_text_source_still_ends(make_harness):
    harness = make_harness()
    errors: list[llm.RealtimeModelError] = []
    harness.session.on("error", errors.append)

    async def _text():
        yield "Please "
        raise RuntimeError("the text source broke")

    await harness.session.say(_text())
    await _settle()

    frames = harness.frames()
    assert _reply_text(frames) == "Please "
    # the reply is closed off, so Fish Audio finishes it and the generation can end
    assert frames[-1]["last"] is True
    assert len(errors) == 1
    assert errors[0].recoverable is True


async def test_generate_reply_follows_its_instructions(make_harness):
    harness = make_harness(_response("Greet the user.", "Hello there."))

    await harness.session.update_instructions("You are a baker.")
    await harness.session.generate_reply(instructions="Greet the user.")
    await _settle()

    assert _reply_text(harness.frames()) == "Hello there."


async def test_update_chat_ctx_keeps_what_the_framework_has_not_seen(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.done(reply_id, _ANSWER)

    # built before the reply ended: it has the user turn but not the reply
    stale = llm.ChatContext.empty()
    stale.items.append(harness.session.chat_ctx.items[0])
    await harness.session.update_chat_ctx(stale)
    assert [item.role for item in harness.session.chat_ctx.items] == ["user", "assistant"]

    # once the framework has sent the reply back, removing it removes it
    await harness.session.update_chat_ctx(harness.session.chat_ctx)
    await harness.session.update_chat_ctx(stale)
    assert [item.role for item in harness.session.chat_ctx.items] == ["user"]


async def test_update_chat_ctx_can_drop_old_turns(make_harness):
    harness = make_harness(_response(_QUESTION, _ANSWER))

    harness.eager()
    await _settle()
    reply_id = harness.frames()[0]["reply_id"]
    harness.committed()
    harness.done(reply_id, _ANSWER)
    await asyncio.sleep(10)

    # the turn is no longer new, so a summary that leaves it out replaces it
    summary = llm.ChatContext.empty()
    summary.add_message(role="assistant", content="They asked about opening hours.")
    await harness.session.update_chat_ctx(summary)

    items = harness.session.chat_ctx.items
    assert len(items) == 1
    assert items[0].text_content == "They asked about opening hours."


async def test_update_chat_ctx_keeps_system_messages(make_harness):
    harness = make_harness()

    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content="The caller is a returning customer.")
    await harness.session.update_chat_ctx(chat_ctx)

    assert harness.session.chat_ctx.items[0].role == "system"


async def test_session_start_frame(make_harness):
    harness = make_harness(voice="voice-id", own_speech="quiet", eager_turns=False)

    frame = harness.session._session_start_frame()

    assert frame["type"] == "session.start"
    assert frame["voice"] == {"reference_id": "voice-id"}
    assert frame["audio"]["sample_rate"] == SAMPLE_RATE
    assert frame["turns"]["eager"] is False
    assert frame["turns"]["speech_started"] is True
    assert frame["own_speech"] == "quiet"


async def test_reply_written_before_the_session_starts_is_sent_once_it_does(make_harness):
    harness = make_harness()
    harness.session._connected = False

    await harness.session.say("Hello.")
    await _settle()
    assert harness.frames() == []

    harness.event("session.started", session_id="s1", limits={}, text_rules={})
    assert _reply_text(harness.frames()) == "Hello."


def test_split_utf8_respects_the_byte_limit():
    pieces = _split_utf8("héllo wörld", 4)

    assert "".join(pieces) == "héllo wörld"
    for piece in pieces:
        assert len(piece.encode("utf-8")) <= 4


def test_text_rules_prompt_follows_the_session_rules():
    prompt = _text_rules_prompt({"numbers": "spoken", "ack_in_t": False, "avoid": ["colon"]})

    assert "spoken words" in prompt
    assert "acknowledgement" in prompt
    assert "colon" in prompt
    assert "semicolons" not in prompt


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------


def _connecting_session(monkeypatch: pytest.MonkeyPatch, run_connection: Any) -> RealtimeSession:
    monkeypatch.setattr(RealtimeSession, "_run_connection", run_connection)
    conn_options = APIConnectOptions(max_retry=2, retry_interval=0.5)
    model = RealtimeModel(
        llm=FakeLLM(), api_key="test-key", http_session=MagicMock(), conn_options=conn_options
    )
    return model.session()


async def test_session_survives_any_number_of_dropped_connections(monkeypatch):
    connections = 0

    async def _run_connection(self: RealtimeSession) -> None:
        nonlocal connections
        connections += 1
        self._handle_event({"type": "session.started", "session_id": f"s{connections}"})
        await asyncio.sleep(1)
        raise APIConnectionError("Fish Audio closed the session")

    session = _connecting_session(monkeypatch, _run_connection)
    errors: list[llm.RealtimeModelError] = []
    reconnects: list[object] = []
    session.on("error", errors.append)
    session.on("session_reconnected", reconnects.append)

    await asyncio.sleep(10)

    # each connection worked for a while, so each drop gets a fresh set of retries
    assert connections > 4
    for error in errors:
        assert error.recoverable is True
    # the event follows the new session, not the loss of the old one
    assert len(reconnects) == connections - 1
    await session.aclose()


async def test_refused_session_fails_for_good(monkeypatch):
    async def _run_connection(self: RealtimeSession) -> None:
        raise APIStatusError("Fish Audio refused the session", status_code=4400, retryable=False)

    session = _connecting_session(monkeypatch, _run_connection)
    errors: list[llm.RealtimeModelError] = []
    session.on("error", errors.append)

    # asked for before the refusal is known
    greeting = session.say("Hello.")
    await asyncio.sleep(1)

    assert len(errors) == 1
    assert errors[0].recoverable is False
    # nothing asked for afterwards is left waiting
    with pytest.raises(llm.RealtimeError):
        await session.say("Still there?")
    generation = await greeting
    assert await _collect(generation.message_stream) == []
    await session.aclose()
