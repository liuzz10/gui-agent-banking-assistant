// This function sets the chat UI to be collapsed or expanded based on the isCollapsed parameter. (for the ease of voice control)
export function setChatCollapsed(isCollapsed) {
  const root = document.getElementById("chatbot-root");
  const collapseBtn = document.getElementById("collapse-btn");

  if (isCollapsed) {
    root.classList.add("chatbot-collapsed");
    collapseBtn.textContent = "▲";
    sessionStorage.setItem("chatbotCollapsed", "true");
  } else {
    root.classList.remove("chatbot-collapsed");
    collapseBtn.textContent = "▼";
    sessionStorage.setItem("chatbotCollapsed", "false");
  }
}

// To set it up when page loaded (user clicks the collapse button to toggle the chat UI)
export function setupChatCollapse() {
  console.log("Setting up chat collapse functionality");
  const collapseBtn = document.getElementById("collapse-btn");

  // ✅ Default to collapsed if no value stored yet
  let stored = sessionStorage.getItem("chatbotCollapsed");

  // If there's no stored value, default to true (collapsed)
  let isCollapsed = stored === null ? true : stored === "true";

  // Apply visual + store
  setChatCollapsed(isCollapsed);

  // Toggle on click
  collapseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    isCollapsed = !isCollapsed;
    setChatCollapsed(isCollapsed);
  });
}