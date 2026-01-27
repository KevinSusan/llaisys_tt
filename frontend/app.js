let activeId = "";
const conversations = [];

const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const endpointInput = document.getElementById("endpoint");
const maxTokensInput = document.getElementById("max-tokens");
const sendButton = document.getElementById("send");
const sessionList = document.getElementById("session-list");
const newChatButton = document.getElementById("new-chat");

const createLocalId = () => {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const appendBubble = (text, role) => {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
};

const renderChat = (conversation) => {
  chat.innerHTML = "";
  for (const message of conversation.messages) {
    appendBubble(message.text, message.role);
  }
};

const renderSessions = () => {
  sessionList.innerHTML = "";
  for (const convo of conversations) {
    const item = document.createElement("div");
    item.className = `session-item${convo.id === activeId ? " active" : ""}`;
    item.textContent = convo.title || "新对话";
    item.addEventListener("click", () => {
      activeId = convo.id;
      renderSessions();
      renderChat(convo);
    });
    sessionList.appendChild(item);
  }
};

const createConversation = () => {
  const convo = {
    id: createLocalId(),
    serverId: "",
    title: "新对话",
    messages: [],
  };
  conversations.unshift(convo);
  activeId = convo.id;
  renderSessions();
  renderChat(convo);
  return convo;
};

const getActiveConversation = () => {
  let convo = conversations.find((c) => c.id === activeId);
  if (!convo) {
    convo = createConversation();
  }
  return convo;
};

const streamChat = async (payload, bubble, convo) => {
  const res = await fetch(`${endpointInput.value}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: true }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`请求失败：${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      const data = JSON.parse(part.slice(6));
      if (data.session_id && !convo.serverId) {
        convo.serverId = data.session_id;
      }
      if (data.delta) {
        bubble.textContent += data.delta;
      }
      if (data.done) {
        return;
      }
    }
  }
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  const convo = getActiveConversation();
  convo.messages.push({ role: "user", text: prompt });
  appendBubble(prompt, "user");
  promptInput.value = "";
  const assistantBubble = appendBubble("", "assistant");
  convo.messages.push({ role: "assistant", text: "" });

  sendButton.disabled = true;
  const payload = {
    prompt,
    max_new_tokens: Number(maxTokensInput.value) || 128,
  };
  if (convo.serverId) {
    payload.session_id = convo.serverId;
  }

  try {
    await streamChat(payload, assistantBubble, convo);
    convo.messages[convo.messages.length - 1].text = assistantBubble.textContent;
    if (convo.title === "新对话") {
      convo.title = prompt.slice(0, 12);
      renderSessions();
    }
  } catch (err) {
    assistantBubble.textContent = `请求失败：${err.message}`;
    convo.messages[convo.messages.length - 1].text = assistantBubble.textContent;
  } finally {
    sendButton.disabled = false;
  }
});

newChatButton.addEventListener("click", () => {
  createConversation();
});

createConversation();
