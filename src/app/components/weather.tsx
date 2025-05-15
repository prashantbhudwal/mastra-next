"use client";

import { getWeatherInfo } from "../actions";

export function Weather() {
  async function handleSubmit() {
    try {
      const result = await getWeatherInfo("bangalore");
      // Handle the result
      console.log(result);
    } catch {
      console.log("Fuck you");
    }
  }

  return (
    <form action={handleSubmit}>
      <button onClick={handleSubmit}>Get Weather</button>
    </form>
  );
}
