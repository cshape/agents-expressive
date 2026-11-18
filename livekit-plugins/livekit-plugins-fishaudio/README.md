# Fish Audio plugin for LiveKit Agents

Support for voice synthesis and realtime speech-to-text with [Fish Audio](https://fish.audio/).

- Docs: `https://docs.fish.audio/`

## Installation

```bash
pip install livekit-plugins-fishaudio
```

## Prerequisites

Obtain an API key from Fish Audio.

Set the API key as an environment variable:

```
FISH_API_KEY=<your_api_key>
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

# Basic usage with env-based credentials
tts = fishaudio.TTS()

session = AgentSession(
    tts=tts,
    # ... stt, llm, etc.
)
```

### Speech-to-text

`fishaudio.STT` streams audio to Fish Audio's realtime ASR. It works with LiveKit's turn detection:

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

session = AgentSession(
    stt=fishaudio.STT(),
    # ... llm, tts, vad, etc.
)
```

Fish Audio can also detect the end of each user turn itself:

```python
from livekit.agents import AgentSession, TurnHandlingOptions
from livekit.plugins import fishaudio

session = AgentSession(
    stt=fishaudio.STT(),
    turn_handling=TurnHandlingOptions(turn_detection="stt"),
    # ... llm, tts, etc.
)
```

### Realtime (Duplex API)

`fishaudio.realtime.RealtimeModel` runs a conversation on Fish Audio's [Duplex API](https://docs.fish.audio/features/duplex-api), which is in preview. Fish Audio listens, detects turns and interruptions, and speaks. The `llm` you pass writes the replies, so instructions, chat context and tools work as with any LLM:

```python
from livekit.agents import AgentSession
from livekit.plugins import fishaudio

session = AgentSession(
    llm=fishaudio.realtime.RealtimeModel(
        llm="google/gemma-4-31b-it",  # a LiveKit Inference model, or any llm.LLM
    ),
)
```

No `stt`, `tts`, `vad` or turn detector is needed. Fish Audio decides what interrupts the agent, and only the text the user heard is kept in the chat context.
