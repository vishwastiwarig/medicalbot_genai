# main.py
from agent import build_agent
from langchain_core.messages import AIMessage, HumanMessage

def run_chat():
    """Starts the interactive chat loop for the agent."""
    agent = build_agent()
    chat_history = []
    
    print("🚀 Agent is ready. Ask a medical question.")
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() == 'exit':
            break
        
        final_state = agent.invoke({
            "question": user_question,
            "chat_history": chat_history
        })
        
        final_answer = final_state['chat_history'][-1].content
        print("\nAgent:", final_answer)
        chat_history = final_state['chat_history']

if __name__ == "__main__":
    run_chat()