import { useCallback, useEffect, useState } from "react";
import type { Message } from "./api";

export interface Chat {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
}

const KEY = "sunbeam.chats.v1";

export const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);

function load(): Chat[] {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as Chat[]) : [];
  } catch {
    return [];
  }
}

export function useChats() {
  const [chats, setChats] = useState<Chat[]>(load);
  const [activeId, setActiveId] = useState<string | null>(() => load()[0]?.id ?? null);

  useEffect(() => {
    try {
      // Keep the 40 most recent conversations.
      localStorage.setItem(KEY, JSON.stringify(chats.slice(0, 40)));
    } catch {
      /* storage full or blocked: chats simply aren't persisted */
    }
  }, [chats]);

  const active = chats.find((c) => c.id === activeId) ?? null;

  const newChat = useCallback(() => setActiveId(null), []);

  const ensureChat = useCallback((firstMessage: string): string => {
    const id = uid();
    const title = firstMessage.length > 48 ? firstMessage.slice(0, 46).trimEnd() + "…" : firstMessage;
    setChats((cs) => [{ id, title, messages: [], updatedAt: Date.now() }, ...cs]);
    setActiveId(id);
    return id;
  }, []);

  const updateMessages = useCallback((chatId: string, fn: (m: Message[]) => Message[]) => {
    setChats((cs) => {
      const next = cs.map((c) => (c.id === chatId ? { ...c, messages: fn(c.messages), updatedAt: Date.now() } : c));
      return next.sort((a, b) => b.updatedAt - a.updatedAt);
    });
  }, []);

  const deleteChat = useCallback(
    (id: string) => {
      setChats((cs) => cs.filter((c) => c.id !== id));
      if (id === activeId) setActiveId(null);
    },
    [activeId],
  );

  return { chats, active, activeId, setActiveId, newChat, ensureChat, updateMessages, deleteChat };
}
