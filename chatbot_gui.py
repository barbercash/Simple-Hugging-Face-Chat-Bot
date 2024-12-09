import tkinter as tk
from tkinter import scrolledtext
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatbotGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Chatbot")
        master.geometry("500x600")
        master.configure(bg='#f0f0f0')

        # Initialize Chatbot Model
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_history_ids = None

        # Create Chat Display
        self.chat_display = scrolledtext.ScrolledText(
            master, 
            wrap=tk.WORD, 
            width=60, 
            height=20, 
            font=('Helvetica', 10),
            state='disabled'
        )
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Create User Input Entry
        self.user_input = tk.Entry(
            master, 
            width=50, 
            font=('Helvetica', 10)
        )
        self.user_input.grid(row=1, column=0, padx=10, pady=10)
        self.user_input.bind('<Return>', self.send_message)

        # Send Button
        self.send_button = tk.Button(
            master, 
            text="Send", 
            command=self.send_message,
            bg='#4CAF50', 
            fg='white'
        )
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        # Welcome Message
        self.display_message("🤖 Chatbot: Hello! I'm ready to chat.")

    def send_message(self, event=None):
        # Get user input
        user_message = self.user_input.get()
        if not user_message.strip():
            return

        # Display user message
        self.display_message(f"You: {user_message}", is_user=True)
        
        # Clear input field
        self.user_input.delete(0, tk.END)

        # Generate response in a separate thread
        threading.Thread(target=self.get_bot_response, args=(user_message,), daemon=True).start()

    def get_bot_response(self, user_input):
        try:
            # Encode user input
            new_user_input_ids = self.tokenizer.encode(
                user_input + self.tokenizer.eos_token, 
                return_tensors='pt'
            )
            
            # Manage chat history
            if self.chat_history_ids is not None:
                bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids
            
            # Generate response
            self.chat_history_ids = self.model.generate(
                bot_input_ids, 
                max_length=1000, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode response
            response = self.tokenizer.decode(
                self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                skip_special_tokens=True
            )
            
            # Display bot response
            self.display_message(f"🤖 Chatbot: {response}")

        except Exception as e:
            self.display_message(f"🤖 Chatbot: Sorry, something went wrong. {str(e)}")

    def display_message(self, message, is_user=False):
        # Enable text widget for editing
        self.chat_display.configure(state='normal')
        
        # Insert message
        self.chat_display.insert(tk.END, message + '\n')
        
        # Scroll to the end
        self.chat_display.see(tk.END)
        
        # Disable text widget
        self.chat_display.configure(state='disabled')

def main():
    root = tk.Tk()
    chatbot_gui = ChatbotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()