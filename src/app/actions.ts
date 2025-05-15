"use server";

import { mastra } from "@/mastra";

export async function getWeatherInfo(city: string) {
  console.log("I started");
  const agent = mastra.getAgent("weatherAgent");
  console.log("I created an agent");

  const result = await agent.generate(`What's the weather like in ${city}?`);
  console.log("I got the result: ", result.text);

  return result.text;
}
