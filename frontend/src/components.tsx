import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ArrowUp,
  BookOpen,
  CalendarDays,
  Check,
  Copy,
  ExternalLink,
  MapPin,
  MessageSquare,
  Moon,
  Plus,
  RotateCcw,
  Sparkles,
  Square,
  Sun,
  Trash2,
  TriangleAlert,
  X,
} from "lucide-react";
import type { Info, Message } from "./api";
import type { Chat } from "./useChats";

/* ---------- brand ---------- */

export function SunMark({ size = 32, spin = false }: { size?: number; spin?: boolean }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <rect width="64" height="64" rx="16" fill="#1B2236" />
      <g className={spin ? "sun-spin" : undefined} stroke="#F5A524" strokeWidth="4" strokeLinecap="round">
        <path d="M32 9v6M32 49v6M9 32h6M49 32h6M15.7 15.7l4.2 4.2M44.1 44.1l4.2 4.2M15.7 48.3l4.2-4.2M44.1 19.9l4.2-4.2" />
      </g>
      <circle cx="32" cy="32" r="10" fill="#F5A524" />
    </svg>
  );
}

function timeAgo(iso: string | null): string {
  if (!iso) return "never";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  const d = Math.round(s / 86400);
  return d === 1 ? "yesterday" : `${d} days ago`;
}

function IconButton(props: { label: string; onClick: () => void; children: ReactNode; className?: string }) {
  return (
    <button
      type="button"
      aria-label={props.label}
      title={props.label}
      onClick={props.onClick}
      className={`grid size-9 place-items-center rounded-lg text-muted transition-colors hover:bg-sun-soft hover:text-ink focus-visible:outline-2 focus-visible:outline-sun ${props.className ?? ""}`}
    >
      {props.children}
    </button>
  );
}

/* ---------- sidebar ---------- */

export function Sidebar(props: {
  chats: Chat[];
  activeId: string | null;
  info: Info | null;
  open: boolean;
  dark: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onToggleTheme: () => void;
}) {
  const { chats, activeId, info } = props;
  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-black/40 transition-opacity md:hidden ${props.open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={props.onClose}
      />
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-line bg-surface transition-transform duration-200 md:static md:translate-x-0 ${
          props.open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center gap-3 px-4 pt-5 pb-4">
          <SunMark size={34} />
          <div className="min-w-0 flex-1">
            <div className="font-display text-[1.05rem] font-semibold leading-tight">Sunbeam</div>
            <div className="text-xs text-muted">Institute assistant</div>
          </div>
          <IconButton label="Close menu" onClick={props.onClose} className="md:hidden">
            <X size={18} />
          </IconButton>
        </div>

        <div className="px-3">
          <button
            type="button"
            onClick={props.onNew}
            className="flex h-10 w-full items-center gap-2 rounded-xl border border-line bg-raised px-3 text-sm font-semibold shadow-[0_1px_0_rgba(0,0,0,0.03)] transition-colors hover:border-sun/60 focus-visible:outline-2 focus-visible:outline-sun"
          >
            <Plus size={16} className="text-sun" />
            New chat
            <kbd className="ml-auto rounded border border-line px-1.5 text-[10px] font-medium text-faint">Ctrl K</kbd>
          </button>
        </div>

        <nav className="scroll-thin mt-4 flex-1 overflow-y-auto px-3 pb-3" aria-label="Conversations">
          {chats.length > 0 && <div className="px-2 pb-1.5 text-[11px] font-semibold tracking-wider text-faint uppercase">Recent</div>}
          {chats.length === 0 && (
            <p className="px-2 py-6 text-center text-xs leading-relaxed text-faint">
              Your conversations will appear here.
            </p>
          )}
          <ul className="space-y-0.5">
            {chats.map((c) => (
              <li key={c.id} className="group relative">
                <button
                  type="button"
                  onClick={() => props.onSelect(c.id)}
                  className={`flex w-full items-center gap-2.5 rounded-lg py-2 pr-9 pl-2.5 text-left text-sm transition-colors ${
                    c.id === activeId ? "bg-sun-soft font-medium text-ink" : "text-muted hover:bg-sun-soft/60 hover:text-ink"
                  }`}
                >
                  <MessageSquare size={15} className={c.id === activeId ? "shrink-0 text-sun" : "shrink-0 opacity-60"} />
                  <span className="truncate">{c.title}</span>
                </button>
                <button
                  type="button"
                  aria-label={`Delete "${c.title}"`}
                  title="Delete chat"
                  onClick={() => props.onDelete(c.id)}
                  className="absolute top-1/2 right-1.5 grid size-7 -translate-y-1/2 place-items-center rounded-md text-faint opacity-0 transition hover:bg-raised hover:text-red-500 focus-visible:opacity-100 group-hover:opacity-100"
                >
                  <Trash2 size={14} />
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="border-t border-line p-3">
          <div className="flex items-center gap-2 rounded-xl bg-paper px-3 py-2.5">
            <span className={`size-2 shrink-0 rounded-full ${info?.ready ? "bg-emerald-500" : "bg-faint"}`} />
            <div className="min-w-0 flex-1 text-xs leading-snug">
              <div className="font-semibold">{info?.ready ? `${info.courses} courses indexed` : "Connecting…"}</div>
              <div className="truncate text-muted">Website data synced {timeAgo(info?.updated_at ?? null)}</div>
            </div>
            <IconButton label={props.dark ? "Switch to light theme" : "Switch to dark theme"} onClick={props.onToggleTheme}>
              {props.dark ? <Sun size={17} /> : <Moon size={17} />}
            </IconButton>
          </div>
        </div>
      </aside>
    </>
  );
}

/* ---------- empty state ---------- */

const SUGGESTIONS = [
  { icon: CalendarDays, title: "Upcoming batches", prompt: "Which courses have batches starting soon, and what are their fees?" },
  { icon: Sparkles, title: "AI & GenAI programs", prompt: "Compare Sunbeam's AI Engineering and Generative AI programs." },
  { icon: BookOpen, title: "Core Java syllabus", prompt: "What is covered in the Core Java course syllabus?" },
  { icon: MapPin, title: "Branches & timings", prompt: "Where are Sunbeam's branches and what are their office timings?" },
];

const CHIPS = ["All courses", "PreCAT preparation", "C-DAC PG programmes", "Internships", "Contact details"];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="m-auto w-full max-w-3xl px-4 py-10">
      <div className="rise">
        <SunMark size={52} spin />
        <h1 className="mt-6 font-display text-3xl leading-tight font-semibold tracking-tight sm:text-[2.6rem]">
          Ask anything about <span className="text-sun-ink">Sunbeam</span>.
        </h1>
        <p className="mt-3 max-w-xl text-[0.95rem] leading-relaxed text-muted">
          Courses, syllabus, batch dates, fees, internships, C-DAC programmes and branches. Answers come from
          Sunbeam's live website data, refreshed every day.
        </p>
      </div>

      <div className="rise mt-8 grid gap-3 sm:grid-cols-2" style={{ animationDelay: "60ms" }}>
        {SUGGESTIONS.map(({ icon: Icon, title, prompt }) => (
          <button
            key={title}
            type="button"
            onClick={() => onPick(prompt)}
            className="group flex items-start gap-3 rounded-2xl border border-line bg-surface p-4 text-left transition hover:-translate-y-0.5 hover:border-sun/50 hover:shadow-[0_8px_24px_-12px_rgba(232,137,12,0.35)] focus-visible:outline-2 focus-visible:outline-sun"
          >
            <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-sun-soft text-sun-ink transition group-hover:bg-sun group-hover:text-white">
              <Icon size={18} />
            </span>
            <span>
              <span className="block text-sm font-semibold">{title}</span>
              <span className="mt-0.5 block text-[0.83rem] leading-snug text-muted">{prompt}</span>
            </span>
          </button>
        ))}
      </div>

      <div className="rise mt-5 flex flex-wrap gap-2" style={{ animationDelay: "120ms" }}>
        {CHIPS.map((c) => (
          <button
            key={c}
            type="button"
            onClick={() => onPick(`Tell me about ${c.toLowerCase()} at Sunbeam.`)}
            className="rounded-full border border-line bg-surface px-3.5 py-1.5 text-[0.8rem] font-medium text-muted transition hover:border-sun/50 hover:text-ink"
          >
            {c}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------- messages ---------- */

function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
        table: ({ children }) => (
          <div className="table-wrap">
            <table>{children}</table>
          </div>
        ),
      }}
    >
      {text}
    </ReactMarkdown>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <IconButton
      label={done ? "Copied" : "Copy answer"}
      onClick={() => {
        navigator.clipboard.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1500);
        });
      }}
      className="size-8"
    >
      {done ? <Check size={15} className="text-emerald-500" /> : <Copy size={15} />}
    </IconButton>
  );
}

export function MessageView(props: {
  message: Message;
  streaming: boolean;
  status: string | null;
  canRetry: boolean;
  onRetry: () => void;
}) {
  const { message: m, streaming, status } = props;

  if (m.role === "user") {
    return (
      <div className="rise flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[#1b2236] px-4 py-2.5 text-[0.95rem] leading-relaxed whitespace-pre-wrap text-white dark:bg-raised dark:text-ink">
          {m.content}
        </div>
      </div>
    );
  }

  const waiting = streaming && !m.content;
  return (
    <div className="rise flex gap-3">
      <div className="mt-0.5 shrink-0">
        <SunMark size={30} spin={streaming} />
      </div>
      <div className="min-w-0 flex-1">
        {waiting && (
          <div className="flex h-8 items-center text-sm font-medium">
            <span className="shimmer">{status ?? "Thinking"}…</span>
          </div>
        )}
        {m.content && (
          <div className={`prose-chat ${streaming ? "caret" : ""}`}>
            <Markdown text={m.content} />
          </div>
        )}
        {m.error && (
          <div className="mt-2 flex items-start gap-2.5 rounded-xl border border-red-300/60 bg-red-50 px-3.5 py-3 text-sm text-red-800 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-200">
            <TriangleAlert size={17} className="mt-0.5 shrink-0" />
            <div className="flex-1">{m.error}</div>
            {props.canRetry && (
              <button
                type="button"
                onClick={props.onRetry}
                className="inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-0.5 font-semibold hover:bg-red-100 dark:hover:bg-red-500/20"
              >
                <RotateCcw size={14} /> Retry
              </button>
            )}
          </div>
        )}
        {!streaming && m.sources && m.sources.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {m.sources.map((s) => (
              <a
                key={s.url + s.title}
                href={s.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-1 text-xs font-medium text-muted transition hover:border-sun/60 hover:text-ink"
              >
                <ExternalLink size={12} className="shrink-0 text-sun" />
                <span className="truncate">{s.title}</span>
              </a>
            ))}
          </div>
        )}
        {!streaming && m.content && (
          <div className="mt-1 -ml-1.5 flex">
            <CopyButton text={m.content} />
            {props.canRetry && (
              <IconButton label="Regenerate answer" onClick={props.onRetry} className="size-8">
                <RotateCcw size={15} />
              </IconButton>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- composer ---------- */

export function Composer(props: { busy: boolean; onSend: (text: string) => void; onStop: () => void; focusKey: string }) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, [props.focusKey]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [text]);

  const submit = () => {
    const t = text.trim();
    if (!t || props.busy) return;
    props.onSend(t);
    setText("");
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:pb-5">
      <div className="flex items-end gap-2 rounded-2xl border border-line bg-raised p-2 pl-4 shadow-[0_10px_30px_-18px_rgba(27,34,54,0.35)] transition focus-within:border-sun/60">
        <label htmlFor="ask" className="sr-only">
          Ask about Sunbeam
        </label>
        <textarea
          id="ask"
          ref={ref}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="Ask about courses, fees, batches, branches…"
          className="max-h-[200px] min-h-[40px] flex-1 resize-none bg-transparent py-2 text-[0.95rem] leading-6 outline-none placeholder:text-faint"
        />
        {props.busy ? (
          <button
            type="button"
            onClick={props.onStop}
            aria-label="Stop answering"
            title="Stop"
            className="grid size-10 shrink-0 place-items-center rounded-xl bg-ink text-paper transition hover:opacity-85"
          >
            <Square size={14} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!text.trim()}
            aria-label="Send message"
            title="Send (Enter)"
            className="grid size-10 shrink-0 place-items-center rounded-xl bg-sun text-white transition hover:brightness-105 disabled:bg-line disabled:text-faint"
          >
            <ArrowUp size={18} strokeWidth={2.5} />
          </button>
        )}
      </div>
      <p className="mt-2 text-center text-[11px] text-faint">
        Answers are generated from sunbeaminfo.in and sunbeaminfo.com. Confirm fees and dates with the institute.
      </p>
    </div>
  );
}
