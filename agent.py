# agent.py

from typing import List, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Import our tools and shared components from other files
from tools import pubmed_search_tool, rag_retriever_tool, web_scraper_tool, osm_location_search_tool
from config import llm

class AgentState(TypedDict):
    """
    Defines the state of our agent. This is the memory that gets passed
    between each step of the graph.
    """
    chat_history: List[BaseMessage]
    question: str
    tool_output: str
    location: dict  # For location-based tools

def router_node(state: AgentState):
    """
    Analyzes the user's question to decide which tool to use. This is the
    agent's main decision-making brain.
    """
    print("---ROUTER: Deciding which tool to use---")
    
    # This detailed prompt is the "source code" for the router's decision.
    prompt = f"""You are an expert routing agent. Based on the user's question, you must decide which of the available tools is the most appropriate to use. You must respond with ONLY the name of the tool.

The available tools are:
1.  **PubMed_Search**: Choose this for questions about the latest scientific research, clinical trials, specific drug studies, or very recent medical findings.
2.  **Internal_Knowledge_Base**: Choose this for questions about well-established medical facts, standard procedures, common disease symptoms, and general definitions found in medical textbooks.
3.  **Web_Scraper**: Choose this for general health questions, patient-friendly explanations, or topics that might not be in a formal textbook (e.g., "what does an MRI feel like?").
4.  **OSM_Location_Search**: Choose this for questions about medical facility locations, hospitals, clinics, or any location-based queries.

User Question: "{state['question']}"
"""
    
    # Call the LLM to get the decision
    response = llm.invoke(prompt)
    decision = response.content.strip()
    print(f"---ROUTER: Decision is '{decision}'---")
    
    # Return a dict with the chosen tool for routing
    if "OSM_Location_Search" in decision:
        return {"next_action": "OSM_Location_Search", **state}
    elif "PubMed_Search" in decision:
        return {"next_action": "PubMed_Search", **state}
    elif "Web_Scraper" in decision:
        return {"next_action": "Web_Scraper", **state}
    else: # Default to the internal knowledge base for safety
        return {"next_action": "Internal_Knowledge_Base", **state}

def generate_answer_node(state: AgentState):
    """
    Generates the final response to the user based on the tool's output
    and the conversation history.
    """
    print("---Generating Final Answer---")
    
    # Compose context with location info if available
    context = state["tool_output"]
    if state.get("location") and state["location"]:
        loc = state["location"]
        context += f"\n\nLocation Info:\nName: {loc.get('display_name')}\nLatitude: {loc.get('lat')}\nLongitude: {loc.get('lon')}"

    # This prompt defines the agent's final persona.
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert medical communicator. Answer the user's question based on the provided CONTEXT.\n"
            "Explain the answer in a simple, easy-to-understand, and friendly tone for a patient.\n"
            "Always end with a disclaimer: 'This is for informational purposes only. Please consult a doctor for medical advice.'\n\n"
            "CONTEXT:\n{context}"
        )),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    answer = chain.invoke({
        "question": state["question"],
        "context": context, # Use the output from the chosen tool and location
        "chat_history": state["chat_history"]
    }).content

    # Update the chat history with the full exchange
    new_history = state["chat_history"] + [HumanMessage(content=state["question"]), AIMessage(content=answer)]
    return {"chat_history": new_history}

    """Builds and compiles the LangGraph agent."""
    workflow = StateGraph(AgentState)

    # Add the nodes to the graph
    workflow.add_node("Router", router_node)
    workflow.add_node("PubMed_Search", pubmed_search_tool)
    workflow.add_node("Internal_Knowledge_Base", rag_retriever_tool)
    workflow.add_node("Web_Scraper", web_scraper_tool)
    workflow.add_node("OSM_Location_Search", osm_location_search_tool)
    workflow.add_node("Generate_Answer", generate_answer_node)

    # Define the edges of the graph
    workflow.set_entry_point("Router")

    # The conditional edge from the router to the appropriate tool
    workflow.add_conditional_edges(
        "Router",
        lambda state: state["next_action"],
        {
            "PubMed_Search": "PubMed_Search",
            "Internal_Knowledge_Base": "Internal_Knowledge_Base",
            "Web_Scraper": "Web_Scraper",
            "OSM_Location_Search": "OSM_Location_Search",
        }
    )

    # Edges from each tool back to the final answer generation node
    workflow.add_edge("PubMed_Search", "Generate_Answer")
    workflow.add_edge("Internal_Knowledge_Base", "Generate_Answer")
    workflow.add_edge("Web_Scraper", "Generate_Answer")
    workflow.add_edge("OSM_Location_Search", "Generate_Answer")

    # The final answer generation node is the end of the line
    workflow.add_edge("Generate_Answer", END)

    # Compile the graph into a runnable agent
    agent = workflow.compile()
    print("✅ Intelligent Router Agent Compiled.")
    return agent
