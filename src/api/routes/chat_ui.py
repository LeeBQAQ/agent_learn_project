from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

CHAT_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#1a1a2e;color:#eee;height:100vh;display:flex;flex-direction:column}
.header{background:#16213e;padding:16px 24px;border-bottom:1px solid #0f3460}
.header h1{font-size:18px;color:#e94560}
.chat{flex:1;overflow-y:auto;padding:24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:80%;padding:12px 16px;border-radius:12px;line-height:1.6;white-space:pre-wrap}
.msg.user{align-self:flex-end;background:#0f3460}
.msg.assistant{align-self:flex-start;background:#16213e}
.msg .label{font-size:12px;color:#e94560;margin-bottom:4px}
.msg .sources{margin-top:8px;font-size:12px;color:#888;border-top:1px solid #333;padding-top:8px}
.input-area{display:flex;padding:16px 24px;gap:12px;background:#16213e;border-top:1px solid #0f3460}
.input-area input{flex:1;padding:12px;border-radius:8px;border:1px solid #0f3460;background:#1a1a2e;color:#eee;font-size:14px}
.input-area button{padding:12px 24px;border:none;border-radius:8px;background:#e94560;color:#fff;cursor:pointer;font-size:14px}
.input-area button:hover{opacity:.9}
.input-area button:disabled{opacity:.5;cursor:not-allowed}
.status{font-size:12px;color:#888;padding:8px 24px;text-align:center}
.upload-area{display:flex;align-items:center;gap:8px;padding:8px 24px;background:#16213e}
.upload-area input[type=file]{display:none}
.upload-area label{padding:6px 14px;border:1px dashed #e94560;border-radius:6px;color:#e94560;cursor:pointer;font-size:12px}
.upload-area label:hover{background:#0f3460}
.upload-area .file-tag{font-size:12px;color:#888;background:#0f3460;padding:4px 10px;border-radius:4px;display:none;align-items:center;gap:8px}
.upload-area .file-tag .del{cursor:pointer;color:#e94560}
.spinner{display:inline-block;width:12px;height:12px;border:2px solid #e94560;border-top-color:transparent;border-radius:50%;animation:spin .8s linear infinite;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="header"><h1>RAG Chat</h1></div>
<div class="chat" id="chat"></div>
<div class="status" id="status"></div>
<div class="upload-area">
  <label for="fileInput">上传文档</label>
  <input type="file" id="fileInput" multiple accept=".txt,.md,.csv,.pdf,.py,.json" onchange="uploadFiles()">
  <span class="file-tag" id="fileTag"><span class="name"></span><span class="del" onclick="clearFile()">x</span></span>
</div>
<div class="input-area">
  <input id="query" placeholder="输入问题，回车发送..." onkeydown="if(event.key==='Enter')ask()">
  <button id="sendBtn" onclick="ask()">发送</button>
</div>
<script>
let sessionId = null;

function addMsg(role, content) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.innerHTML = '<div class="label">' + (role==="user"?"你":"AI") + '</div>' + content;
  document.getElementById("chat").appendChild(div);
  document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
}

async function uploadFiles() {
  const input = document.getElementById("fileInput");
  const files = input.files;
  if (!files.length) return;

  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("collection", "default");

  document.getElementById("status").innerHTML = '<span class="spinner"></span>上传中...';
  try {
    const resp = await fetch("/api/v1/documents/upload", {method: "POST", body: form});
    const data = await resp.json();
    document.getElementById("status").textContent = `已上传 ${data.total} 个文件`;
    const tag = document.getElementById("fileTag");
    tag.querySelector(".name").textContent = Array.from(files).map(f=>f.name).join(", ");
    tag.style.display = "inline-flex";
  } catch(e) {
    document.getElementById("status").textContent = "上传失败: " + e.message;
  }
  input.value = "";
}

function clearFile() {
  document.getElementById("fileInput").value = "";
  document.getElementById("fileTag").style.display = "none";
}

async function ask() {
  const input = document.getElementById("query");
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  input.disabled = true;
  document.getElementById("sendBtn").disabled = true;
  document.getElementById("status").innerHTML = '<span class="spinner"></span>思考中...';

  addMsg("user", query);
  const aiDiv = document.createElement("div");
  aiDiv.className = "msg assistant";
  aiDiv.innerHTML = '<div class="label">AI</div><span class="content"></span><div class="sources"></div>';
  document.getElementById("chat").appendChild(aiDiv);
  const contentEl = aiDiv.querySelector(".content");
  const sourcesEl = aiDiv.querySelector(".sources");

  try {
    const resp = await fetch("/api/v1/query/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, session_id: sessionId})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const parts = buffer.split("\\n\\n");
      buffer = parts.pop();
      for (const part of parts) {
        const lines = part.split("\\n");
        const event = lines[0].replace("event: ", "");
        const data = JSON.parse(lines[1].replace("data: ", ""));
        switch (event) {
          case "status":
            document.getElementById("status").innerHTML = '<span class="spinner"></span>' + data.message;
            break;
          case "sources":
            if (data.length) sourcesEl.innerHTML = "参考: " + data.map(s=>s.source).join(", ");
            break;
          case "token":
            contentEl.textContent += data.content;
            document.getElementById("chat").scrollTop = document.getElementById("chat").scrollHeight;
            break;
          case "done":
            sessionId = sessionId || data.session_id;
            document.getElementById("status").textContent = "置信度: " + (data.confidence*100).toFixed(0) + "%";
            break;
          case "error":
            contentEl.textContent += "\\n[错误: " + data.message + "]";
            document.getElementById("status").textContent = "出错了";
            break;
        }
      }
    }
  } catch (e) {
    contentEl.textContent += "\\n[连接失败: " + e.message + "]";
    document.getElementById("status").textContent = "连接失败";
  } finally {
    input.disabled = false;
    document.getElementById("sendBtn").disabled = false;
  }
}
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(content=CHAT_HTML)
