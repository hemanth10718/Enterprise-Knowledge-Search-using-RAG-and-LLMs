import React, { useState } from "react";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendMessage = async () => {
    if (!input.trim()) return;

    // Add user message to chat UI
    const newMessages = [...messages, { sender: "user", text: input }];
    setMessages(newMessages);

    try {
      const res = await fetch(
        `http://127.0.0.1:8000/chat/?user_query=${encodeURIComponent(input)}&top_k=3`,
        { method: "POST" }
      );
      const data = await res.json();

      // Format backend response
      const aiText =
        data.results && data.results.length > 0
          ? data.results
              .map(
                (r) =>
                  `📄 ${r.filename} (Rank ${r.rank})\n   Snippet: ${
                    r.citations && r.citations.length > 0
                      ? r.citations.join(" | ")
                      : "No snippet"
                  }`
              )
              .join("\n\n")
          : "No matching resumes found.";

      setMessages([
        ...newMessages,
        { sender: "ai", text: aiText },
      ]);
    } catch (err) {
      console.error("Error:", err);
      setMessages([
        ...newMessages,
        { sender: "ai", text: "⚠️ Error connecting to backend." },
      ]);
    }

    setInput(""); // clear input
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h2>💬 Resume Chat (React + FastAPI)</h2>

      <div
        style={{
          border: "1px solid #ccc",
          padding: "10px",
          height: "400px",
          overflowY: "auto",
          marginBottom: "10px",
          background: "#f9f9f9",
        }}
      >
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              textAlign: m.sender === "user" ? "right" : "left",
              margin: "8px 0",
            }}
          >
            <span
              style={{
                display: "inline-block",
                padding: "10px",
                borderRadius: "10px",
                background: m.sender === "user" ? "#007bff" : "#e4e6eb",
                color: m.sender === "user" ? "white" : "black",
                maxWidth: "70%",
                whiteSpace: "pre-line",
              }}
            >
              {m.text}
            </span>
          </div>
        ))}
      </div>

      <input
        type="text"
        value={input}
        placeholder="Ask something about resumes..."
        onChange={(e) => setInput(e.target.value)}
        style={{ width: "70%", marginRight: "10px", padding: "8px" }}
        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
      />
      <button onClick={sendMessage} style={{ padding: "8px 15px" }}>
        Send
      </button>
    </div>
  );
}

export default App;
