import os
import json
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, trim_messages
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, List, Optional, Literal, Any, Dict, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, MessagesState, START, END, add_messages
from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
import operator

load_dotenv()

# Ensure the GOOGLE_API_KEY environment variable is set.
if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# --- 1. CONFIGURE LLM ---
try:
    # We'll use a fast and capable model for these structured tasks
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
except Exception as e:
    print(f"Error initializing the LLM. Please ensure your GOOGLE_API_KEY is set correctly.")
    print(e)
    exit()

# --- 2. AGENT PROMPTS ---

AGENT_PROMPTS = {
    "ReasoningPlanner": """
    You are the Reasoning Planner in a multi-agent system designed for LLM Context Engineering.
    Analyze the user's input context and any pre-existing issues. Determine which specialized agents must be activated to improve the context.

    Available Agents:
    - Write: Optimizes instructions and clarity.
    - Select: Ensures relevance of retrieved knowledge.
    - Compress: Assesses and reduces context length.
    - Isolate: Recommends partitioning context to prevent overload.
    - Poisoning: Scans for inaccuracies, errors, or hallucinations.
    - Distraction: Identifies and removes irrelevant information.
    - Clash: Finds and flags contradictory information.

    Today's Date is: Friday, August 8, 2025.

    User Input Context:
    ---
    {input_context}
    ---
    Previously Identified Issues: {context_issues}

    Your task: Based on the context and issues, create an execution plan.
    - Prioritize agents that address critical issues (Poisoning, Clash) first.
    - Then, add general optimization agents (Write, Select, Compress) if applicable.

    Respond with ONLY a JSON object containing a single key 'plan', which is a list of agent names to be executed in order.
    Example: {{"plan": ["Poisoning", "Clash", "Write"]}}
    """,
    "Write": """
    You are the 'Write' Agent. Your role is to optimize the clarity and structure of the context.
    Analyze the following context and provide actionable recommendations.

    Input Context:
    ---
    {input_context}
    ---

    Focus on:
    1.  Clarity and specificity of instructions and goals.
    2.  Effectiveness of the prompt structure.
    3.  Organization of working memory or scratchpad sections.

    Respond with ONLY a JSON object containing two keys: 'analysis_results' (a list of strings describing your findings) and 'recommendations' (a list of strings with your suggestions).
    Example: {{"analysis_results": ["The main task is not clearly stated."], "recommendations": ["Rewrite the task to start with 'Your primary goal is...'."]}}
    """,
    # The prompts for Select, Compress, Isolate, Poisoning, Distraction, and Clash follow the same JSON output structure.
    "Select": """You are the 'Select' Agent. Your role is to ensure only the most relevant knowledge and data are included in the context. Analyze the following context:
    ---
    {input_context}
    ---
    Respond with ONLY a JSON object containing 'analysis_results' and 'recommendations' lists. Focus on data relevance and necessity.""",
    "Compress": """You are the 'Compress' Agent. Your role is to optimize context length by identifying redundancies. Analyze the following context:
    ---
    {input_context}
    ---
    Respond with ONLY a JSON object containing 'analysis_results' and 'recommendations' lists. Suggest summarization or removal of verbose sections.""",
    "Isolate": """You are the 'Isolate' Agent. Your role is to recommend strategies for context isolation. Analyze the following context:
    ---
    {input_context}
    ---
    Respond with ONLY a JSON object containing 'analysis_results' and 'recommendations' lists. Focus on partitioning distinct tasks or states.""",
    "Poisoning": """You are the 'Poisoning Detector'. Scan the context for hallucinations, factual inaccuracies, or explicit errors. Analyze the following context:
    ---
    {input_context}
    ---
    Respond with ONLY a JSON object containing 'analysis_results' and 'recommendations' lists. Highlight any poisoned or erroneous information.""",
    "Distraction": """You are the 'Distraction Filter'. Identify irrelevant information that could distract the main LLM. Analyze the following context:
    ---
    {input_context}
    ---
    Respond with ONLY a JSON object containing 'analysis_results' and 'recommendations' lists. Identify off-topic or low-value information for pruning.""",
    "Clash": """You are the 'Clash Detector'. Identify contradictory information within the context. Analyze the following context:
    ---
    {input_context}
    ---
    Respond with ONLY a JSON object containing 'analysis_results' and 'recommendations' lists. Pinpoint conflicting instructions or data points.""",
    "Guardian": """
    You are the Guardian Agent. Review the collected analysis and recommendations to provide a final quality assessment.

    Analysis Results:
    ---
    {analysis_results}
    ---
    Recommendations:
    ---
    {recommendations}
    ---

    Your task: Determine if there are any remaining CRITICAL issues that require another round of planning and refinement. Critical issues typically involve data integrity (from Poisoning) or logical conflicts (from Clash).

    Respond with ONLY a JSON object containing a single key 'context_issues'. This should be a list of strings describing any critical issues found. If the context is approved and ready for synthesis, return an empty list.
    Example (issues found): {{"context_issues": ["Critical data error flagged by Poisoning agent must be re-evaluated."]}}
    Example (approved): {{"context_issues": []}}
    """,
    "Synthesizer": """
    You are the Synthesizer Agent. Your task is to rewrite the original context by applying the provided recommendations.

    Original Context:
    ---
    {input_context}
    ---
    Actionable Recommendations:
    ---
    {recommendations}
    ---

    Your sole output must be the final, optimized context as a single string. Do not add any commentary, greetings, or other text.
    """
}

# --- 3. LANGGRAPH AGENT STATE ---

class AgentState(TypedDict):
    input_context: str
    optimized_context: str
    analysis_results: Annotated[List[str], operator.add]
    recommendations: Annotated[List[str], operator.add]
    context_issues: Annotated[List[str], operator.add]
    plan: List[str]
    next_agent: str

# --- 4. NODE DEFINITIONS ---

def create_agent_chain(prompt_template: str, parser):
    """Helper function to create a LangChain execution chain."""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm | parser

def reasoning_planner(state: AgentState):
    print("--- 🧠 Reasoning Planner ---")
    planner_chain = create_agent_chain(AGENT_PROMPTS["ReasoningPlanner"], JsonOutputParser())
    
    result = planner_chain.invoke({
        "input_context": state["input_context"],
        "context_issues": state["context_issues"] or "None"
    })
    
    plan = result.get("plan", [])
    print(f"Plan: {plan}")
    return {"plan": plan, "next_agent": plan[0] if plan else None}

def agent_executor(state: AgentState):
    agent_name = state["next_agent"]
    print(f"--- 🛠️ Agent Executor: {agent_name} ---")
    
    # All analysis agents use a JSON parser
    agent_chain = create_agent_chain(AGENT_PROMPTS[agent_name], JsonOutputParser())
    result = agent_chain.invoke({"input_context": state["input_context"]})
    
    print(f"Analysis: {result.get('analysis_results')}")
    print(f"Recommendations: {result.get('recommendations')}")

    # Update the plan
    current_plan = state["plan"][:]
    if agent_name in current_plan:
        current_plan.remove(agent_name)
    next_agent = current_plan[0] if current_plan else None
    
    # Merge agent results with plan updates
    result["plan"] = current_plan
    result["next_agent"] = next_agent
    return result

def guardian_agent(state: AgentState):
    print("--- 🛡️ Guardian Agent ---")
    guardian_chain = create_agent_chain(AGENT_PROMPTS["Guardian"], JsonOutputParser())
    
    # We pass the accumulated analysis/recommendations as a string
    result = guardian_chain.invoke({
        "analysis_results": "\n".join(state["analysis_results"]),
        "recommendations": "\n".join(state["recommendations"])
    })
    
    issues = result.get("context_issues", [])
    print(f"Guardian check complete. Issues found: {len(issues)}")
    if issues:
        print(f"Issues: {issues}")
        # Clear previous analysis for the refinement loop
        return {"context_issues": issues, "analysis_results": [], "recommendations": []}
    return {"context_issues": issues}


def synthesizer(state: AgentState):
    print("--- ✨ Synthesizer ---")
    synthesizer_chain = create_agent_chain(AGENT_PROMPTS["Synthesizer"], StrOutputParser())
    
    optimized_context = synthesizer_chain.invoke({
        "input_context": state["input_context"],
        "recommendations": "\n".join(state["recommendations"])
    })
    print("Optimized context generated.")
    return {"optimized_context": optimized_context}

# --- 5. GRAPH DEFINITION & EDGES ---

def route_agents(state: AgentState) -> str:
    """Routes from agent executor back to itself or to the guardian."""
    return "continue" if state["next_agent"] else "guardian"

def route_guardian(state: AgentState) -> str:
    """Routes from the guardian to the planner for refinement or to the synthesizer."""
    return "refine" if state["context_issues"] else "synthesize"

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("planner", reasoning_planner)
workflow.add_node("agent_executor", agent_executor)
workflow.add_node("guardian", guardian_agent)
workflow.add_node("synthesizer", synthesizer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "agent_executor")

# Conditional edge for the agent execution loop
workflow.add_conditional_edges(
    "agent_executor",
    route_agents,
    {"continue": "agent_executor", "guardian": "guardian"}
)

# Conditional edge for the refinement loop
workflow.add_conditional_edges(
    "guardian",
    route_guardian,
    {"refine": "planner", "synthesize": "synthesizer"}
)

workflow.add_edge("synthesizer", END)

# Compile the graph into a runnable application
build_graph = workflow.compile()

# --- 6. EXAMPLE CONTEXT (TESTS) ---
#plan: Plan = llm.with_structured_output(Plan).invoke(planner_prompt.format(task=state["task"], plan=state["plan"]))
initial_context = """
Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries
"""