import { mastra } from "@/mastra";

export const maxDuration = 30;

export async function POST(req: Request) {
  const request = await req.json();
  console.log(request);
  const { messages, system, tools, runConfig } = request;

  console.log(runConfig);
  console.log(messages);
  console.log(tools);
  console.log(system);

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
