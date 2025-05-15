import { mastra } from "@/mastra";
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  const { city } = await req.json();
  console.log("server 1", city);
  const agent = mastra.getAgent("weatherAgent");

  console.log("server 2", city);

  const result = await agent.stream(`What's the weather like in ${city}?`);

  console.log("server 3", result);

  return result.toDataStreamResponse();
}
