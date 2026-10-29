"""Interactive chat with conversation memory (``policy-rag chat``, ``rag.chat()``)."""
from __future__ import annotations

from pathlib import Path

HELP = """Commands: /new (start a new conversation)  /doc <id ...> | /doc all (choose documents)
          /save [file] (save the conversation as JSON)  /history  /help  /quit"""


def run_chat(rag, conversation, input_fn=input, print_fn=print, save_dir: str = "runs/chats"):
    """Ask questions in a loop; each answer sees the conversation so far through the question rewriter."""
    conv = conversation
    docs = ", ".join(rag.corpus.documents)
    if rag.rewriter is not None:
        memory = (f"Memory on: follow-up questions are rewritten from the last {rag.cfg.memory.max_turns} turns "
                  "(experimental, not adopted).")
    else:
        memory = ("Memory off: every question is answered as typed, so name what you ask about. To try follow-ups, "
                  "start with --set memory.enabled=true (experimental).")
    print_fn(f"Chat about: {docs}. {memory}\n{HELP}")
    while True:
        try:
            q = input_fn("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit", "/q"):
            break
        if q == "/help":
            print_fn(HELP)
            continue
        if q == "/new":
            conv = rag.new_conversation(conv.documents)
            print_fn("New conversation.")
            continue
        if q.startswith("/doc"):
            ids = q.split()[1:]
            try:
                conv.documents = None if not ids or ids == ["all"] else list(rag.new_conversation(ids).documents)
                print_fn(f"Documents: {', '.join(conv.documents) if conv.documents else 'all'}")
            except KeyError as e:
                print_fn(f"  {e}")
            continue
        if q.startswith("/save"):
            parts = q.split(maxsplit=1)
            path = Path(parts[1]) if len(parts) > 1 else Path(save_dir) / f"{conv.id}.json"
            print_fn(f"Saved {conv.save(path)} (it contains policy text: keep it private)")
            continue
        if q == "/history":
            for n, t in enumerate(conv.turns, 1):
                print_fn(f"{n}. {t.question}" + (f"  [searched as: {t.standalone}]" if t.standalone != t.question else ""))
            continue
        rag.ask(q, conversation=conv)
    return conv
