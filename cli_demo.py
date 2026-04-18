#!/usr/bin/env python3
import sys
from inference import run

def main():
    history = []
    
    print("Tool-Calling Assistant CLI")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear history")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
            
            if prompt.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break
            
            if prompt.lower() == "clear":
                history = []
                print("History cleared.")
                continue
            
            if not prompt:
                continue
            
            response = run(prompt, history)
            
            print(f"\nAssistant: {response}")
            
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()