import { mastra } from "@/mastra";
import { gemini } from "@/mastra/models";
import { openai } from "@ai-sdk/openai";
import { frontendTools } from "@assistant-ui/react-ai-sdk";
import { streamText } from "ai";

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages, system, tools } = await req.json();
  const myAgent = mastra.getAgent("weatherAgent");
  const stream = await myAgent.stream(messages);

  //   const result = streamText({
  //     model: gemini,
  //     messages,
  //     toolCallStreaming: true,
  //     system,
  //     tools: {
  //       ...frontendTools(tools),
  //       // add backend tools here
  //     },
  //   });

  return stream.toDataStreamResponse();
}
