import customtkinter as ctk
import tkinter as tk
import requests
import threading
import queue
import json

# --- Configuration ---
LLAMA_API_URL = "http://127.0.0.1:8080/completion"

# --- Main Application Class ---
class LlamaGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Llama.cpp GUI")
        self.geometry("1000x700")

        # --- Data ---
        self.response_queue = queue.Queue()
        self.is_generating = False

        # --- Layout ---
        self.grid_columnconfigure(0, weight=1, minsize=250)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Frame (Controls) ---
        self.controls_frame = ctk.CTkFrame(self, width=250)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.controls_frame.grid_propagate(False) # Prevent frame from resizing to fit widgets

        # Add widgets to the control frame
        self.create_control_widgets()

        # --- Right Frame (Chat) ---
        self.chat_frame = ctk.CTkFrame(self)
        self.chat_frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nswe")
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        # Add widgets to the chat frame
        self.create_chat_widgets()

        # Start checking the queue for responses
        self.after(100, self.process_queue)

    def create_control_widgets(self):
        """Creates all the sliders and inputs for controlling the model."""
        # Title
        title_label = ctk.CTkLabel(self.controls_frame, text="Model Parameters", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=(10, 20), padx=10)

        # Temperature
        temp_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        temp_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(temp_frame, text="Temperature").pack(side="left")
        self.temp_slider = ctk.CTkSlider(temp_frame, from_=0.0, to=2.0, number_of_steps=200)
        self.temp_slider.pack(side="right", fill="x", expand=True, padx=(10,0))
        self.temp_slider.set(0.8)

        # Top-P
        top_p_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        top_p_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(top_p_frame, text="Top-P").pack(side="left")
        self.top_p_slider = ctk.CTkSlider(top_p_frame, from_=0.0, to=1.0, number_of_steps=100)
        self.top_p_slider.pack(side="right", fill="x", expand=True, padx=(10,0))
        self.top_p_slider.set(0.95)

        # Repeat Penalty
        repeat_penalty_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        repeat_penalty_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(repeat_penalty_frame, text="Repeat Penalty").pack(side="left")
        self.repeat_penalty_slider = ctk.CTkSlider(repeat_penalty_frame, from_=1.0, to=1.5, number_of_steps=50)
        self.repeat_penalty_slider.pack(side="right", fill="x", expand=True, padx=(10,0))
        self.repeat_penalty_slider.set(1.1)
        
        # Max Tokens (n_predict)
        max_tokens_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        max_tokens_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(max_tokens_frame, text="Max Tokens").pack(side="left")
        self.max_tokens_entry = ctk.CTkEntry(max_tokens_frame)
        self.max_tokens_entry.pack(side="right", fill="x", expand=True, padx=(10,0))
        self.max_tokens_entry.insert(0, "1024")

        # GPU Layers (Note: This cannot be changed after server start)
        gpu_layers_frame = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        gpu_layers_frame.pack(fill="x", padx=10, pady=20)
        ctk.CTkLabel(gpu_layers_frame, text="GPU Layers (Restart Server to Change)", wraplength=230).pack()

        # Status Label
        self.status_label = ctk.CTkLabel(self.controls_frame, text="Ready", text_color="gray")
        self.status_label.pack(side="bottom", pady=10)


    def create_chat_widgets(self):
        """Creates the chat history box, user input field, and send button."""
        self.chat_history = ctk.CTkTextbox(self.chat_frame, state="disabled", font=("Arial", 14), wrap="word")
        self.chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.prompt_entry = ctk.CTkEntry(self.chat_frame, placeholder_text="Type your message...", font=("Arial", 14), height=40)
        self.prompt_entry.grid(row=1, column=0, padx=(10, 5), pady=(0, 10), sticky="ew")
        self.prompt_entry.bind("<Return>", self.send_message)

        self.send_button = ctk.CTkButton(self.chat_frame, text="Send", command=self.send_message, height=40)
        self.send_button.grid(row=1, column=1, padx=(5, 10), pady=(0, 10), sticky="ew")

    def send_message(self, event=None):
        """Handles sending the prompt to the backend."""
        if self.is_generating:
            return

        prompt = self.prompt_entry.get().strip()
        if not prompt:
            return

        self.is_generating = True
        self.send_button.configure(state="disabled", text="Generating...")
        self.status_label.configure(text="Generating response...")

        self.update_chat_history(f"You: {prompt}\n\n")
        self.prompt_entry.delete(0, "end")

        # Start generation in a separate thread to not freeze the GUI
        thread = threading.Thread(target=self.get_model_response, args=(prompt,))
        thread.daemon = True
        thread.start()
        
    def get_model_response(self, prompt):
        """Worker thread function to call the Llama.cpp server."""
        try:
            payload = {
                "prompt": prompt,
                "stream": True,
                "temperature": self.temp_slider.get(),
                "top_p": self.top_p_slider.get(),
                "repeat_penalty": self.repeat_penalty_slider.get(),
                "n_predict": int(self.max_tokens_entry.get())
            }

            self.response_queue.put(("ASSISTANT_START", "Assistant: "))
            
            with requests.post(LLAMA_API_URL, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_data = json.loads(decoded_line[6:])
                            content = json_data.get("content", "")
                            self.response_queue.put(("TOKEN", content))
                            if json_data.get("stop"):
                                break
        except requests.exceptions.RequestException as e:
            self.response_queue.put(("ERROR", f"\n\n[Error: {e}])"))
        except Exception as e:
            self.response_queue.put(("ERROR", f"\n\n[An unexpected error occurred: {e}])"))
        finally:
            self.response_queue.put(("DONE", "\n\n"))

    def process_queue(self):
        """Processes messages from the response queue to update the GUI."""
        try:
            while not self.response_queue.empty():
                msg_type, data = self.response_queue.get_nowait()
                if msg_type == "ASSISTANT_START":
                    self.update_chat_history(data)
                elif msg_type == "TOKEN":
                    self.update_chat_history(data, end="")
                elif msg_type == "ERROR":
                    self.update_chat_history(data, end="\n\n")
                    self.reset_ui_after_generation()
                elif msg_type == "DONE":
                    self.reset_ui_after_generation()

        except queue.Empty:
            pass
        finally:
            # Reschedule the queue check
            self.after(100, self.process_queue)

    def update_chat_history(self, text, end="\n"):
        """Appends text to the chat history box."""
        self.chat_history.configure(state="normal")
        self.chat_history.insert("end", text + end)
        self.chat_history.configure(state="disabled")
        self.chat_history.see("end") # Auto-scroll

    def reset_ui_after_generation(self):
        """Resets the UI elements to their default state."""
        self.is_generating = False
        self.send_button.configure(state="normal", text="Send")
        self.status_label.configure(text="Ready")

# --- Run the Application ---
if __name__ == "__main__":
    ctk.set_appearance_mode("dark") # "dark", "light", "system"
    app = LlamaGUI()
    app.mainloop()
