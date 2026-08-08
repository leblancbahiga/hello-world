/* NURU Lite — client minimal en vanilla JS.
   WebSocket pour le streaming chat, REST pour les listes et documents. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const els = {
    navBtns: [...document.querySelectorAll(".nav-btn")],
    convList: $("conv-list"),
    messages: $("messages"),
    input: $("input"),
    send: $("send"),
    stop: $("stop"),
    title: $("conv-title"),
    meta: $("meta"),
    status: $("status"),
    modelSelect: $("model-select"),
    ragCheck: $("rag-check"),
    docList: $("doc-list"),
    fileInput: $("file-input"),
    settingsMsg: $("settings-msg"),
  };

  const state = {
    ws: null,
    view: "chat",
    conversationId: null,
    generating: false,
    conversations: [],
    assistantBody: null, // corps de la réponse en cours de streaming
  };

  // ------------------------------------------------------------- helpers

  function setStatus(text, online) {
    els.status.textContent = text;
    els.status.className = "status" + (online ? " online" : "");
  }

  function addMessage(role, content) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    if (role !== "user") {
      const tag = document.createElement("span");
      tag.className = "role-tag";
      tag.textContent = role === "assistant" ? "NURU" : "erreur";
      div.appendChild(tag);
    }
    const body = document.createElement("div");
    body.textContent = content;
    div.appendChild(body);
    els.messages.appendChild(div);
    els.messages.scrollTop = els.messages.scrollHeight;
    return body;
  }

  function addSources(sources) {
    if (!sources || !sources.length) return;
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${sources.length})`;
    const ul = document.createElement("ul");
    for (const src of sources) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "src-name";
      name.textContent = src.document + (src.page ? ` — p.${src.page}` : "");
      const snippet = document.createElement("div");
      snippet.textContent = (src.content || "").slice(0, 180) + "…";
      li.appendChild(name);
      li.appendChild(snippet);
      ul.appendChild(li);
    }
    details.appendChild(summary);
    details.appendChild(ul);
    els.messages.appendChild(details);
    els.messages.scrollTop = els.messages.scrollHeight;
  }

  function clearMessages() { els.messages.innerHTML = ""; }

  function showEmpty() {
    clearMessages();
    const div = document.createElement("div");
    div.className = "empty";
    div.innerHTML =
      "<strong>NURU Lite</strong><br>Assistant personnel local, offline-first.<br>" +
      "Écrivez un message pour commencer. Cochez « Documents » pour chercher dans vos fichiers indexés.";
    els.messages.appendChild(div);
  }

  function renderConversations() {
    els.convList.innerHTML = "";
    for (const conv of state.conversations) {
      const item = document.createElement("div");
      item.className = "conv-item" + (conv.id === state.conversationId ? " active" : "");
      item.textContent = conv.title;
      item.onclick = () => openConversation(conv.id);
      els.convList.appendChild(item);
    }
  }

  function updateComposer() {
    els.send.disabled = state.generating;
    els.stop.hidden = !state.generating;
  }

  function setMeta(model, provider) {
    els.meta.textContent = (model && provider) ? `${provider} · ${model}` : "";
  }

  function switchView(view) {
    state.view = view;
    for (const btn of els.navBtns) {
      btn.classList.toggle("active", btn.dataset.view === view);
    }
    for (const el of document.querySelectorAll(".view")) {
      el.classList.toggle("active", el.id === view);
    }
    els.convList.style.display = view === "chat" ? "" : "none";
    if (view === "docs") loadDocuments();
    if (view === "settings") loadSettings();
  }

  // -------------------------------------------------------------- réseau

  async function api(path, opts) {
    const resp = await fetch(path, opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || resp.status);
    return data;
  }

  async function loadConversations() {
    const data = await api("/api/conversations");
    state.conversations = data.items;
    renderConversations();
  }

  async function loadModels() {
    try {
      const data = await api("/api/models");
      els.modelSelect.innerHTML = "";
      for (const group of data.items) {
        if (!group.models.length) continue;
        const optgroup = document.createElement("optgroup");
        optgroup.label = group.provider;
        for (const m of group.models) {
          const opt = document.createElement("option");
          opt.value = `${group.provider}:${m}`;
          opt.textContent = m;
          optgroup.appendChild(opt);
        }
        els.modelSelect.appendChild(optgroup);
      }
      const { provider, model } = data.defaults;
      els.modelSelect.value = `${provider}:${model}`;
    } catch (e) {
      console.warn("models:", e);
    }
  }

  // ---------------------------------------------------------- documents

  function docStatusBadge(status) {
    const span = document.createElement("span");
    span.className = "badge " + status;
    span.textContent = status;
    return span;
  }

  async function loadDocuments() {
    let data;
    try {
      data = await api("/api/documents");
    } catch (e) {
      return;
    }
    els.docList.innerHTML = "";
    const items = data.items || [];
    if (!items.length) {
      const div = document.createElement("div");
      div.className = "empty-list";
      div.textContent = "Aucun document indexé. Cliquez sur « Importer des fichiers… ».";
      els.docList.appendChild(div);
      return;
    }
    for (const doc of items) {
      const row = document.createElement("div");
      row.className = "doc-row";
      const name = document.createElement("div");
      name.className = "doc-name";
      name.textContent = doc.name;
      name.title = doc.name;
      const meta = document.createElement("div");
      meta.className = "doc-meta";
      meta.textContent =
        `${(doc.size / 1024).toFixed(0)} Ko · ${doc.chunks} chunks` +
        (doc.error ? ` · ${doc.error}` : "");
      const del = document.createElement("button");
      del.className = "doc-del";
      del.textContent = "Supprimer";
      del.onclick = async () => {
        await api(`/api/documents/${doc.id}`, { method: "DELETE" });
        loadDocuments();
      };
      row.appendChild(name);
      row.appendChild(docStatusBadge(doc.status));
      row.appendChild(meta);
      row.appendChild(del);
      els.docList.appendChild(row);
    }
    // actualisation pendant l'indexation
    const busy = items.some((d) => d.status === "indexing" || d.status === "pending");
    clearTimeout(loadDocuments._timer);
    if (busy) loadDocuments._timer = setTimeout(loadDocuments, 1500);
  }

  async function importFiles(files) {
    if (!files.length) return;
    const form = new FormData();
    for (const file of files) form.append("files", file);
    await api("/api/documents/import", { method: "POST", body: form });
    loadDocuments();
  }

  // -------------------------------------------------------------- settings

  async function loadSettings() {
    const s = await api("/api/settings");
    $("set-system").value = s.llm?.system_prompt || "";
    $("set-provider").value = s.llm?.default_provider || "";
    $("set-model").value = s.llm?.default_model || "";
    $("set-temp").value = s.llm?.temperature ?? "";
    $("set-embedder").value = s.rag?.embedder || "hash";
    $("set-embedder-model").value = s.rag?.embedder_model || "";
    $("set-topk").value = s.rag?.top_k ?? 5;
  }

  async function saveSettings() {
    els.settingsMsg.textContent = "";
    try {
      await api("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          llm: {
            system_prompt: $("set-system").value,
            default_provider: $("set-provider").value.trim(),
            default_model: $("set-model").value.trim(),
            temperature: parseFloat($("set-temp").value) || 0.7,
          },
          rag: {
            embedder: $("set-embedder").value,
            embedder_model: $("set-embedder-model").value.trim(),
            top_k: parseInt($("set-topk").value, 10) || 5,
          },
        }),
      });
      els.settingsMsg.textContent = "Paramètres enregistrés.";
      loadModels();
    } catch (e) {
      els.settingsMsg.textContent = "Erreur : " + e.message;
    }
  }

  // -------------------------------------------------------------- chat ws

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/ws/chat`);
    state.ws = ws;

    ws.onopen = () => setStatus("connecté", true);
    ws.onclose = () => {
      setStatus("déconnecté — reconnexion…", false);
      state.generating = false;
      updateComposer();
      setTimeout(connect, 1500);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => handleMessage(JSON.parse(event.data));
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "history":
        state.conversationId = msg.conversation_id;
        clearMessages();
        if (!msg.messages || !msg.messages.length) { showEmpty(); return; }
        for (const m of msg.messages) addMessage(m.role, m.content);
        els.title.textContent = msg.title || "—";
        els.messages.scrollTop = els.messages.scrollHeight;
        break;
      case "start":
        state.generating = true;
        updateComposer();
        setMeta(msg.model, msg.provider);
        els.title.textContent = msg.title || els.title.textContent;
        state.conversationId = msg.conversation_id;
        state.assistantBody = addMessage("assistant", "");
        break;
      case "delta":
        if (state.assistantBody) state.assistantBody.textContent += msg.content;
        els.messages.scrollTop = els.messages.scrollHeight;
        break;
      case "done":
      case "stopped":
        if (state.assistantBody) {
          state.assistantBody.remove();
          state.assistantBody = null;
        }
        addSources(msg.sources);
        state.generating = false;
        updateComposer();
        loadConversations();
        break;
      case "error":
        if (state.assistantBody) {
          state.assistantBody.textContent = msg.message;
          state.assistantBody.parentElement.className = "msg error";
          state.assistantBody = null;
        } else {
          addMessage("error", msg.message);
        }
        state.generating = false;
        updateComposer();
        break;
    }
  }

  function send() {
    const content = els.input.value.trim();
    if (!content || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    addMessage("user", content);
    els.input.value = "";
    els.input.style.height = "auto";
    const [provider, model] = els.modelSelect.value.split(":") || [];
    state.ws.send(JSON.stringify({
      type: "send",
      conversation_id: state.conversationId,
      content,
      model: model || "",
      provider: provider || "",
      use_rag: els.ragCheck.checked,
    }));
  }

  function stop() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  function newChat() {
    state.conversationId = null;
    els.title.textContent = "—";
    setMeta("", "");
    showEmpty();
    loadConversations();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "open" }));
    }
  }

  function openConversation(id) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "open", conversation_id: id }));
    }
  }

  // ------------------------------------------------------------- events

  for (const btn of els.navBtns) {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  }
  $("new-chat").addEventListener("click", newChat);
  $("import-btn").addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => {
    importFiles(els.fileInput.files);
    els.fileInput.value = "";
  });
  $("save-settings").addEventListener("click", saveSettings);

  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  els.input.addEventListener("input", () => {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 180) + "px";
  });
  els.send.addEventListener("click", send);
  els.stop.addEventListener("click", stop);
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "n") { e.preventDefault(); newChat(); }
  });

  // -------------------------------------------------------------- boot

  loadModels();
  loadConversations();
  showEmpty();
  connect();
})();
