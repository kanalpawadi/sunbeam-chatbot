import { useCallback, useEffect, useRef, useState } from "react";
import { Menu, Plus } from "lucide-react";
import { fetchInfo, streamChat, type Info, type Message } from "./api";
import { Composer, EmptyState, MessageView, Sidebar, SunMark } from "./components";
import { uid, useChats } from "./useChats";

export default function App() {
  const { chats, active, activeId, setActiveId, newChat, ensureChat, updateMessages, deleteChat } = useChats();
  const [info, setInfo] = useState<Info | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const [busy, setBusy] = useState<{ chatId: string; msgId: string; status: string | null } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  useEffect(() => {
    const load = () => fetchInfo().then(setInfo).catch(() => setInfo(null));
    load();
    const t = setInterval(load, 5 * 60_000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try {
      localStorage.setItem("theme", dark ? "dark" : "light");
    } catch {
      /* ignore */
    }
  }, [dark]);

  // Ctrl/Cmd+K starts a new chat
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        newChat();
        setSidebarOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [newChat]);

  // Follow the stream unless the user scrolled up
  const messages = active?.messages ?? [];
  const lastContent = messages[messages.length - 1]?.content;
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages.length, lastContent, busy?.status]);

  const run = useCallback(
    async (chatId: string, history: { role: "user" | "assistant"; content: string }[]) => {
      const msgId = uid();
      updateMessages(chatId, (m) => [...m, { id: msgId, role: "assistant", content: "" }]);
      setBusy({ chatId, msgId, status: null });
      stickToBottom.current = true;
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      const patch = (fn: (m: Message) => Message) =>
        updateMessages(chatId, (ms) => ms.map((m) => (m.id === msgId ? fn(m) : m)));

      try {
        await streamChat(
          history,
          (ev) => {
            if (ev.type === "token") patch((m) => ({ ...m, content: m.content + ev.text }));
            else if (ev.type === "status") setBusy((b) => (b ? { ...b, status: ev.text } : b));
            else if (ev.type === "sources") patch((m) => ({ ...m, sources: ev.items }));
            else if (ev.type === "error") patch((m) => ({ ...m, error: ev.message }));
          },
          ctrl.signal,
        );
      } catch (err) {
        if (ctrl.signal.aborted) {
          patch((m) => (m.content ? m : { ...m, content: "_Stopped._" }));
        } else {
          const msg = err instanceof Error ? err.message : String(err);
          patch((m) => ({
            ...m,
            error: msg.includes("fetch") ? "Can't reach the Sunbeam server. Check that the backend is running." : msg,
          }));
        }
      } finally {
        abortRef.current = null;
        setBusy(null);
      }
    },
    [updateMessages],
  );

  const send = useCallback(
    (text: string) => {
      if (busy) return;
      const chatId = active?.id ?? ensureChat(text);
      const prior = (active?.messages ?? []).filter((m) => !m.error && m.content);
      const user: Message = { id: uid(), role: "user", content: text };
      updateMessages(chatId, (m) => [...m, user]);
      setSidebarOpen(false);
      run(chatId, [...prior, user].map(({ role, content }) => ({ role, content })));
    },
    [active, busy, ensureChat, run, updateMessages],
  );

  const retry = useCallback(() => {
    if (!active || busy) return;
    const ms = active.messages;
    const lastUser = ms.map((m) => m.role).lastIndexOf("user");
    if (lastUser < 0) return;
    const history = ms.slice(0, lastUser + 1).filter((m) => !m.error && m.content);
    updateMessages(active.id, (m) => m.slice(0, lastUser + 1));
    run(active.id, history.map(({ role, content }) => ({ role, content })));
  }, [active, busy, run, updateMessages]);

  return (
    <div className="flex h-full">
      <Sidebar
        chats={chats}
        activeId={activeId}
        info={info}
        open={sidebarOpen}
        dark={dark}
        onClose={() => setSidebarOpen(false)}
        onSelect={(id) => {
          setActiveId(id);
          setSidebarOpen(false);
        }}
        onNew={() => {
          newChat();
          setSidebarOpen(false);
        }}
        onDelete={(id) => {
          if (busy?.chatId === id) abortRef.current?.abort();
          deleteChat(id);
        }}
        onToggleTheme={() => setDark((d) => !d)}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-2 border-b border-line px-3 md:hidden">
          <button
            type="button"
            aria-label="Open menu"
            onClick={() => setSidebarOpen(true)}
            className="grid size-10 place-items-center rounded-lg text-muted hover:bg-sun-soft"
          >
            <Menu size={20} />
          </button>
          <SunMark size={26} />
          <span className="font-display font-semibold">Sunbeam Assistant</span>
          <button
            type="button"
            aria-label="New chat"
            onClick={newChat}
            className="ml-auto grid size-10 place-items-center rounded-lg text-muted hover:bg-sun-soft"
          >
            <Plus size={20} />
          </button>
        </header>

        <div
          ref={scrollRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
          className="scroll-thin flex flex-1 flex-col overflow-y-auto"
        >
          {messages.length === 0 ? (
            <EmptyState onPick={send} />
          ) : (
            <div className="mx-auto w-full max-w-3xl space-y-7 px-4 pt-8 pb-6">
              {messages.map((m, i) => (
                <MessageView
                  key={m.id}
                  message={m}
                  streaming={busy?.msgId === m.id}
                  status={busy?.msgId === m.id ? busy.status : null}
                  canRetry={!busy && i === messages.length - 1 && m.role === "assistant"}
                  onRetry={retry}
                />
              ))}
            </div>
          )}
        </div>

        <Composer
          busy={!!busy}
          onSend={send}
          onStop={() => abortRef.current?.abort()}
          focusKey={activeId ?? "new"}
        />
      </main>
    </div>
  );
}
