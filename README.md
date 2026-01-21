# Réunion-zzz
I tried a couple of meeting summarization solutions out there. I didn't like them all. I
mostly have meetings in Korean but use a lot of English terms, and they all go wrong in
STT (speech-to-text, transcription). It is kinda inevitable that STT struggles in my use
case. So I just do it all by myself, including manual review/modification of the
transcribed text.

Typical process of making a meeting note.
1. Record audio
2. Transcribe (Use `transcribe_meeting.py` covers it)
3. Review (I do it. It takes max 10m. Not bad, I think.)
    - Diarization
    - Sectioning
4. LLM to summarize (TODO)
    For now, I just copy-paste the whole content in chat UI.


# TODO
https://platform.openai.com/docs/guides/speech-to-text
- Try out auto diarization
- Try out speaker refs
