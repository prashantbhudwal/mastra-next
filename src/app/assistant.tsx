"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime } from "@assistant-ui/react-ai-sdk";
import { Thread } from "@/components/assistant-ui/thread";

export const Assistant = () => {
  const runtime = useChatRuntime({
    api: "/api/chat",
    adapters: {},
    initialMessages: [
      { role: "user", content: "What is your name" },
      { role: "assistant", content: "My name is botter the bot." },
    ],
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  );
};
