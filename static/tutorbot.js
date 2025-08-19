let chatHistory = [];
let intent = null;
let lastSelector = null;
let botMessage = null;
let state = JSON.parse(sessionStorage.getItem("state") || "null");
let speech_rate = 1;
const waitToTakeAction = 5000;
const welcomeMessage = "Hi! I'm Alex. Tell me what you want to do, for example, e-transfer, and I'll walk you through.";
const currentPage = window.parent.location.pathname.split("/").pop();

// This function sets the chat UI to be collapsed or expanded based on the isCollapsed parameter. (for the ease of voice control)
function setChatCollapsed(isCollapsed) {
  const root = document.getElementById("chatbot-root");
  const collapseBtn = document.getElementById("collapse-btn");

  if (isCollapsed) {
    root.classList.add("chatbot-collapsed");
    collapseBtn.textContent = "▲";
    sessionStorage.setItem("chatbotCollapsed", "true");   // ✅ Save to sessionStorage
  } else {
    root.classList.remove("chatbot-collapsed");
    collapseBtn.textContent = "▼";
    sessionStorage.setItem("chatbotCollapsed", "false");  // ✅ Save to sessionStorage
  }
}

// To set it up when page loaded (user clicks the collapse button to toggle the chat UI)
function setupChatCollapse() {
  console.log("Setting up chat collapse functionality");
  const collapseBtn = document.getElementById("collapse-btn");

  // ✅ Default to collapsed if no value stored yet
  let stored = sessionStorage.getItem("chatbotCollapsed");

  // If there's no stored value, default to false (open)
  let isCollapsed = stored === null ? false : stored === "true";

  // Apply visual + store
  setChatCollapsed(isCollapsed);

  // Toggle on click
  collapseBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    isCollapsed = !isCollapsed;
    setChatCollapsed(isCollapsed);
  });
}

function setupKeyboardToggle() {
  const inputSection = document.querySelector(".chatbot-input");
  const toggleBtn = document.getElementById("toggle-input-btn");

  // Default to hidden
  inputSection.style.display = "none";
  toggleBtn.classList.remove("active");

  toggleBtn.addEventListener("click", () => {
    const isVisible = inputSection.style.display !== "none";
    inputSection.style.display = isVisible ? "none" : "flex";

    // Toggle visual active state
    toggleBtn.classList.toggle("active", !isVisible);
  });
}

// This function toggles the listening state of the chatbot.
function toggleListening() {
  const isChecked = document.getElementById("listen-checkbox").checked;
  const statusLabel = document.getElementById("listening-status");

  const activationScreen = document.getElementById("activation-screen");
  const messages = document.getElementById("messages");

  if (isChecked) {
    recognition.start();
    listening = true;
    statusLabel.textContent = "Listening...";
    sessionStorage.setItem("listening", "true");

    // ✅ Show chat messages, hide activation screen
    activationScreen.style.display = "none";
    messages.style.display = "block";

    // 👇 Expand chatbot when turned on
    setChatCollapsed(false);

    appendMessage("assistant", welcomeMessage);

    if (intent) {
      console.log("Calling sendMessage for resuming");
      sendMessage(true);
    }

  } else if (!isChecked && listening) {
    recognition.stop();
    listening = false;
    statusLabel.textContent = "Not listening";
    sessionStorage.setItem("listening", "false");

    // ✅ Show activation screen again, hide chat messages
    activationScreen.style.display = "block";
    messages.style.display = "none";

    // // 👇 Collapse chatbot when turned off
    // setChatCollapsed(true);
  }
}

// Function to get the summary of the last user message based on the current page
// This function is called when the chatbot detects a confirmation or success page.
function getSummary() {
  console.log("getSummary called for page:", currentPage);

  const patterns = {
    "confirm_transfer.html": {
      matches: ["You plan to send"],
      strip: /^\s*✅\s*/    // remove leading check if present
    },
    "success.html": {
      // support both transfer success and payee-added success
      matches: ["You successfully transfered", "Payee added"],
      strip: /^\s*[🎉✅]\s*/ // remove leading 🎉 or ✅
    },
    "payee_added.html": {
      // support both transfer success and payee-added success
      matches: ["Payee added"],
      strip: /^\s*[🎉✅]\s*/ // remove leading 🎉 or ✅
    }
  };

  const pattern = patterns[currentPage];
  if (!pattern) {
    console.log("No pattern defined for this page.");
    return "";
  }

  // Find the most recent user message that contains any of the match phrases
  for (let i = chatHistory.length - 1; i >= 0; i--) {
    const m = chatHistory[i];
    if (m.role !== "user") continue;

    if (pattern.matches.some(txt => m.content.includes(txt))) {
      const cleaned = m.content.replace(pattern.strip, "");
      console.log("Found matching message:", cleaned);
      return cleaned;
    }
  }

  console.log("No matching message found.");
  return "";
}


// Append a message to the chatbox
// This function is called when the user sends a message or the assistant responds.
function appendMessage(role, text, suppressTTS = false) {
    const messages = document.getElementById("messages");
    const div = document.createElement("div");
    div.className = "message " + role;
    div.innerText = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;

    // Speak if the message is from assistant and on the listening mode
    if (role === "assistant" && listening && !suppressTTS) {
    if (currentPage === "confirm_transfer.html" || currentPage === "confirm_bill.html" || currentPage === "payee_added.html" || currentPage === "success.html") {
        const userLog = getSummary();
        if (userLog) {
        speak(userLog);
        }
    }

    speak(text);
    }
}

// Highlight the element in the parent window
// This function is called when the chatbot receives a selector from the backend
let activeHighlights = new Set();

function highlight(selector, lastInstruction = "mark-complete") {
    if (!selector) return;

    // Avoid re-highlighting the same selector
    if (!activeHighlights.has(selector)) {
    console.log("Highlighting:", selector);
    window.parent.postMessage({ selector, instruction: "highlight" }, "*");
    activeHighlights.add(selector);
    }
}

function dehighlightAll(lastInstruction = "mark-complete") {
    activeHighlights.forEach(selector => {
    console.log("Dehighlighting:", selector);
    window.parent.postMessage({ selector, instruction: lastInstruction }, "*");
    });
    activeHighlights.clear();
}

// Log user action to the chat history and sessionStorage
function logUserAction(text) {
    console.log("Logging user action:", text);
    appendMessage("user", text);
    chatHistory.push({ role: "user", content: text });
    sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
}

// Update substep flags for the "Transfer Someone" page
// This function checks the parent form fields to determine if the user has selected an account and entered an amount.
// It returns an object with flags that indicate the progress of the substep.
// This is used to track the user's progress in the transfer process.
function updateSubstepFlagsForTransferSomeone() {
    const substep_flags = {};

    try {
    const parentDoc = window.parent.document;
    const account = parentDoc.querySelector("#from-account");
    const amount = parentDoc.querySelector("#amount");

    if (account && account.value !== "instruction") {
        substep_flags.account_chosen = true;
    }

    if (amount && parseFloat(amount.value) > 0) {
        substep_flags.amount_entered = true;
    }
    } catch (e) {
    console.warn("Unable to access parent form fields:", e);
    }

    return substep_flags;
}

function updateSubstepFlagsForAddPayeeForm() {
  const substep_flags = {};

  try {
    const parentDoc = window.parent.document;
    const payee = parentDoc.querySelector("#payee-name");
    const account = parentDoc.querySelector("#account-number");

    if (payee && payee.value.trim() !== "") {
      substep_flags.name_filled = true;
    }

    if (account && /^\d{11}$/.test(account.value.trim())) {
      substep_flags.account_filled = true;
    }
  } catch (e) {
    console.warn("Unable to access parent form fields:", e);
  }

  return substep_flags;
}




// OPTION1: TTS function using server-side API
// async function speak(text) {
//   try {
//     const response = await fetch("/speak", {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json"
//       },
//       body: JSON.stringify({ text })
//     });

//     const result = await response.json();
//     if (result.status === "success") {
//       console.log("Speech played:", result.text);
//     } else {
//       console.error("TTS Error:", result.reason);
//     }
//   } catch (error) {
//     console.error("Failed to fetch /speak:", error);
//   }
// }

// OPTION2: TTS function using Web Speech API
let voicesReady = false;
let isSpeaking = false;

speechSynthesis.onvoiceschanged = () => {
    voicesReady = true;
    // console.log("Voices loaded:", speechSynthesis.getVoices());
};

function speak(text) {
    if (!('speechSynthesis' in window)) return;
    if (!voicesReady) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    // ✅ Adjust the speech rate here
    utterance.rate = speech_rate; 

    const voices = speechSynthesis.getVoices();
    const voice = voices.find(v => v.lang === 'en-AU' && v.name.includes("Google"));
    if (voice) utterance.voice = voice;

    utterance.onstart = function () {
    isSpeaking = true;
    if (listening && recognition) {
        recognition.abort();  // ⛔️ stop immediately
    }
    };

    utterance.onend = function () {
    isSpeaking = false;
    if (listening && recognition) {
        setTimeout(() => recognition.start(), 800);  // ✅ add delay
    }
    };

    speechSynthesis.speak(utterance);
}

// sendMessage() fires when: 
// 1) The user types a message (e.g., "what's next?") 
// 2) the page auto-resumes on load (newPageLoaded=True)
// 3) a field is updated (e.g., "from account" or "amount" in the transfer process)
async function sendMessage(newPageLoaded = false, overrideTranscript = null) {
    console.log("sendMessage called", { newPageLoaded});
    const input = document.getElementById("chat-input");
    const message = newPageLoaded ? "resuming" : (overrideTranscript ? overrideTranscript : input.value.trim()); // Get user input or use overrideMessage if auto is true.
    console.log("message:", message);
    if (!message && !newPageLoaded) return; // Don't send empty messages unless the page just loaded.
    
    substep_flags = JSON.parse(sessionStorage.getItem("substep_flags") || "{}");
    console.log("substep_flags:", substep_flags); 

    // This is used to track the user's progress in the transfer process.
    if (currentPage === "send_to_alex.html" || currentPage === "pay_bell.html") {
        substep_flags = updateSubstepFlagsForTransferSomeone();
    }
    if (currentPage === "add_payee.html") {
        substep_flags = updateSubstepFlagsForAddPayeeForm();
    }   

    // Append the message to the chat history and display it
    if (!newPageLoaded) {
    appendMessage("user", message);
    chatHistory.push({ role: "user", content: message });
    }

    input.value = "";

    console.log("sending messages to backend:", chatHistory);
    const res = await fetch("/tutorbot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
        messages: chatHistory,
        newPageLoaded,
        intent,
        currentPage,
        substep_flags,  // ✅ Send subtask progress
        assistant: "grace",  // Always use Grace for this chatbot
    })
    });

    const data = await res.json();
    console.log("data from backend", data)

    intent = data.intent
    botMessage = data.botMessage || "";
    
    appendMessage("assistant", botMessage);
    chatHistory.push({ role: "assistant", content: botMessage });
    sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));

    if (intent && intent !== "unknown") {
    sessionStorage.setItem("intent", intent);
    }

    if (data.substep_flags) {
    substep_flags = data.substep_flags;
    sessionStorage.setItem("substep_flags", JSON.stringify(substep_flags));
    }  

    // This checks if the backend returned actions (like "fill", "click", or "select")
    // and sends them to the main page after a short delay.
    // Dehighlight anything from the last step
    dehighlightAll();
    if (Array.isArray(data.action)) {
    data.action.forEach(act => {
        if (act.selector && act.action) {
        highlight(act.selector); // highlight immediately
        }

        if (act.immediate_reply) {
        appendMessage("assistant", act.immediate_reply);
        chatHistory.push({ role: "assistant", content: act.immediate_reply });
        sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
        }
    });
    }
}

// runs only once when chatbot.html is first rendered in the browser (i.e., when the iframe is inserted into the DOM and loads the chatbot page).
window.addEventListener("DOMContentLoaded", () => {
    // Activate collapse logic
    setupChatCollapse(); 

    // Activate keyboard toggle for input
    setupKeyboardToggle();

    // Let parent know which assistant
    console.log("Sending assistant to parent: grace");
    window.parent.postMessage({ instruction: "sendAssistant", assistant: "grace" }, "*");

    // Load chat history, intent
    console.log("DOMContentLoaded");  // 🔍 baseline check
    chatHistory = JSON.parse(sessionStorage.getItem("chatHistory") || "[]");
    intent = sessionStorage.getItem("intent") || null;

    // Restore listening state to show the listening checkbox and status consistently
    const storedListening = sessionStorage.getItem("listening");
    listening = storedListening === "true";

    // ✅ Show/hide activation screen based on listening state (for the following pages)
    const activationScreen = document.getElementById("activation-screen");
    const messages = document.getElementById("messages");
    // If bot is listening, show messages; otherwise, show activation screen
    if (listening) {
    activationScreen.style.display = "none";
    messages.style.display = "block";
    } else {
    activationScreen.style.display = "block";
    messages.style.display = "none";
    }

    document.getElementById("listen-checkbox").checked = listening;
    document.getElementById("listening-status").textContent = listening ? "Listening" : "Not listening";
    if (listening && recognition) {
    recognition.start();
    }

    if (intent) {
    console.log("Calling sendMessage: Auto-resuming on new page");
    sendMessage(true);  // ✅ Clean, unified resume
    }

    // Restore chatbox and history
    for (const m of chatHistory) appendMessage(m.role, m.content, suppressTTS=true); // Re-render chat history
    const input = document.getElementById("chat-input");
    input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        console.log("Calling sendMessage on Enter key");
        sendMessage();
    }
    });

    // Comment this out to disable auto-resume on page load
    // Automatically send a message: if there's no intent, asking for intent. If there's, resuming the conversation.
    // if (!intent) {
    //   appendMessage("assistant", "Hi! I'm Alex, your banking assistant. How can I help you today?");
    // } else {
    //   // Figure out the step based on intent + page
    //   // This logic is only run when the chatbot is first loaded, not on every message sent.
    //   // This is to ensure that the chatbot can resume the conversation from the correct step.
    //   console.log("calling sendMessage for resuming")
    //   sendMessage(newPageLoaded = true);
    // }

    // if (intent) {
    //   // Figure out the step based on intent + page
    //   // This logic is only run when the chatbot is first loaded, not on every message sent.
    //   // This is to ensure that the chatbot can resume the conversation from the correct step.
    //   console.log("calling sendMessage for resuming")
    //   sendMessage(newPageLoaded = true);
    // }

    // Set up event listeners for the parent form fields
    // This is to ensure that the chatbot can update the substep flags when the user selects
    if (window.parent.location.pathname.endsWith("send_to_alex.html") || window.parent.location.pathname.endsWith("pay_bell.html")) {
        let parentDoc = window.parent.document;
        console.log("parentDoc", parentDoc);

        parentDoc.querySelector("#from-account")?.addEventListener("change", () => {
            sendMessage(newPageLoaded = true);
        });

        parentDoc.querySelector("#amount")?.addEventListener("change", () => {
            sendMessage(newPageLoaded = true);
        });
    }

    if (window.parent.location.pathname.endsWith("pay_bell.html")) {
        let parentDoc = window.parent.document;
        parentDoc.querySelector("#auto-pay")?.addEventListener("change", () => {
            sendMessage(newPageLoaded = true);
        });
    }

    // Using change instead of input for the payee form to avoid too many events
    if (window.parent.location.pathname.endsWith("add_payee.html")) {
        let parentDoc = window.parent.document;
        parentDoc.querySelector("#payee-name")?.addEventListener("change", () => {
            sendMessage(newPageLoaded = true);
        });
        
        parentDoc.querySelector("#account-number")?.addEventListener("change", () => {
            sendMessage(newPageLoaded = true);
        });
    }
});

// This listens for messages from the parent window (the main app) to log user actions.
window.addEventListener("message", (event) => {
    const { instruction, text } = event.data;
    if (instruction === "log" && typeof text === "string") {
            console.log("Received log message from parent:", text);
            logUserAction(text);
            // console.log("Calling sendMessage() because of log instruction");
            // sendMessage(true, false, null)
    }
});

// This saves the chat history, step name, and intent to sessionStorage before the page is unloaded.
window.addEventListener("beforeunload", () => {
    sessionStorage.setItem("chatHistory", JSON.stringify(chatHistory));
    sessionStorage.setItem("intent", intent);
});

// Speech Recognition setup
let recognition;
let listening = false;  // global listening flag

if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();  // Chrome
} else if ('SpeechRecognition' in window) {
    recognition = new SpeechRecognition();  // Firefox
}

if (recognition) {
    recognition.continuous = false;  // Not enable continuous listening
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {
    if (isSpeaking) {
        console.warn("Ignoring recognition during TTS");
        return;
    }
    const transcript = event.results[0][0].transcript;
    console.log("You said:", transcript);
    sendMessage(false, transcript);
    };

    recognition.onerror = function(event) {
    if (event.error === "no-speech" || event.error === "aborted") {
        // Suppress benign errors
        console.log("Ignored speech error:", event.error);
        return;
    }

    // Only alert on real unexpected errors
    alert("Speech recognition error: " + event.error);
    };

    recognition.onend = function() {
    if (listening) {
        if (isSpeaking) {
        console.log("TTS still speaking, delaying recognition restart...");
        setTimeout(() => recognition.onend(), 500); // retry until TTS ends
        } else {
        console.log("Restarting recognition...");
        recognition.start();
        }
    }
    };
} else {
    function toggleListening() {
    alert("Speech recognition is not supported in this browser.");
    }
}

// Expose functions to global scope for HTML inline access
window.sendMessage = sendMessage;
window.toggleListening = toggleListening;