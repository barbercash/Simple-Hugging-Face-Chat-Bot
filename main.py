import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        """
        Initialize the chatbot with a pre-trained conversational model
        
        Args:
            model_name (str): Hugging Face model to use for conversation
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Initialize chat history
        self.chat_history_ids = None

    def get_response(self, user_input):
        """
        Generate a response to user input
        
        Args:
            user_input (str): Message from the user
        
        Returns:
            str: Chatbot's response
        """
        # Encode the user input
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, 
            return_tensors='pt'
        )
        
        # Append to chat history
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat([self.chat_history_ids, new_user_input_ids], dim=-1)
        else:
            bot_input_ids = new_user_input_ids
        
        # Generate a response 
        self.chat_history_ids = self.model.generate(
            bot_input_ids, 
            max_length=1000, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the response
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        return response

    def chat(self):
        """
        Interactive chat loop
        """
        print("🤖 Chatbot: Hi there! Type 'quit' to exit.")
        
        while True:
            try:
                user_input = input("You: ")
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🤖 Chatbot: Goodbye!")
                    break
                
                response = self.get_response(user_input)
                print("🤖 Chatbot:", response)
            
            except KeyboardInterrupt:
                print("\n🤖 Chatbot: Chat ended.")
                break

# Run the chatbot
if __name__ == "__main__":
    chatbot = SimpleChatbot()
    chatbot.chat()