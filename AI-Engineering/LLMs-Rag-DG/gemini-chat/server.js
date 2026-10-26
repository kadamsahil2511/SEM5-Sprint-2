import readline from "node:readline/promises";
import "dotenv/config";

const contents = [];

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

console.log("Gemini CLI Chat (type 'exit' to quit)\n");

while (true) {
  const input = await rl.question("You: ");

  if (input.trim().toLowerCase() === "exit") break;
  if (!input.trim()) continue;

  try {
    const userMessage = { role: "user", parts: [{ text: input }] };
    const response = await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": process.env.GEMINI_API_KEY,
      },
      body: JSON.stringify({
        system_instruction: { parts: [{ text: "You are a helpful assistant. Answer clearly and concisely." }] },
        contents: [...contents, userMessage],
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message ?? `Gemini API returned ${response.status}`);

    const modelMessage = data.candidates?.[0]?.content;
    const text = modelMessage?.parts?.filter((part) => !part.thought && part.text).map((part) => part.text).join("");
    if (!text) throw new Error("Gemini returned no text response");

    contents.push(userMessage, modelMessage);

    console.log("\nGemini:", text, "\n");
  } catch (error) {
    console.error("Error:", error.message);
  }
}

rl.close();
