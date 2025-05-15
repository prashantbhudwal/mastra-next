"use server";

import { embedMany } from "ai";
import { openai } from "@ai-sdk/openai";
import similarity from "compute-cosine-similarity";

export async function embedAndCompareProd(text1: string, text2: string) {
  // 1️⃣ Batch-embed both texts
  const { embeddings } = await embedMany({
    model: openai.embedding("text-embedding-3-small"),
    values: [text1, text2],
  });
  const [vec1, vec2] = embeddings;

  // 2️⃣ Compute cosine similarity using the prod library
  const sim = similarity(vec1, vec2);

  return sim;
}
