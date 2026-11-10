import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


def run_terminal_chatbot():
    # 1. Load the modern Qwen2.5-0.5B model and tokenizer
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model_name = "curia007/pick3_llm_model"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Setup hardware acceleration
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device)
    print(f"Model successfully loaded onto: {device}")
    print("=" * 50)
    print("Welcome to your local AI Assistant! Type 'exit' or 'quit' to end.")
    print("=" * 50)

    # 3. Initialize persistent conversation memory with a system prompt
    messages = [
        {"role": "system", "content": "You are a helpful, conversational, and concise terminal assistant."}
    ]

    # 4. Core interaction loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        # Append the new user turn to history
        messages.append({"role": "user", "content": user_input})

        # Apply the chat template sequence
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(device)

        # 5. Setup TextIteratorStreamer and threading
        # We use TextIteratorStreamer so we can easily collect the text as it prints
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,  # Higher cap for ongoing multi-turn dialogue
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )

        # Run generation in a background thread so the loop doesn't block printing
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 6. Stream model responses live and accumulate the string
        print("AI: ", end="", flush=True)
        full_response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_response += new_text

        # Append the model's finished response back to history for next turn memory
        messages.append({"role": "assistant", "content": full_response})
        print()  # Line break after text stream finishes


if __name__ == "__main__":
    run_terminal_chatbot()