# **OpenAI Swarm: Comprehensive Guide to Lightweight Multi-Agent Orchestration (Jan 1 2023 – May 26 2025\)**

## **1\. Introduction and Evolution of OpenAI's Agent Orchestration**

The development of sophisticated Artificial Intelligence (AI) systems capable of performing complex tasks autonomously has accelerated significantly. A key aspect of this evolution is the ability to orchestrate multiple AI agents, allowing them to collaborate and tackle problems that a single agent might find intractable. In this context, OpenAI introduced the Agents SDK, a lightweight, Python-first open-source framework designed to simplify the creation and management of multi-agent workflows.1 This SDK represents a significant step towards enabling developers to build production-ready agentic applications with minimal abstractions.1

The OpenAI Agents SDK evolved from an earlier experimental project known as "Swarm".1 Swarm was conceived as an ergonomic and controllable framework for exploring patterns in multi-agent systems, particularly focusing on concepts like "agents," "handoffs," and "routines" to facilitate collaboration.6 While Swarm was primarily for educational and experimental use, not intended for production settings 7, the Agents SDK builds upon these foundational ideas to offer a more robust, production-ready solution.1 The SDK is provider-agnostic, supporting OpenAI's Responses and Chat Completions APIs, as well as over 100 other Large Language Models (LLMs) through integrations like LiteLLM.3

This report provides a comprehensive guide to the OpenAI Agents SDK, covering its architecture, core components, advanced features, use cases, deployment strategies, and a comparative analysis with other agent orchestration frameworks. The analysis spans developments and information available from January 1, 2023, to May 26, 2025\. The objective is to offer a detailed understanding of the SDK's capabilities and its role in the evolving landscape of multi-agent AI systems.

## **2\. Installation and Initial Setup**

Developing applications with the OpenAI Agents SDK begins with establishing the correct environment and installing the necessary software. This section outlines the prerequisites, installation process, environment configuration, and basic examples to initiate development.

### **2.1 Python Version Requirements**

The OpenAI Agents SDK requires a Python version of 3.8 or higher.9 Some examples and integrations, particularly those involving newer functionalities or dependencies like agents-sdk-models, may specify Python 3.9+.11 It is advisable to use a Python version within this supported range to ensure compatibility and access to all features.

### **2.2 Installation via pip**

The primary method for installing the OpenAI Agents SDK is using pip, the Python package installer. The SDK is available on the Python Package Index (PyPI) under the name openai-agents.

The installation command is:

Bash

pip install openai-agents

3

For functionalities involving voice capabilities, the SDK can be installed with an optional voice dependency group:

Bash

pip install 'openai-agents\[voice\]'

3

To use the SDK with a broad range of LLM providers via LiteLLM, the litellm dependency group can be installed:

Bash

pip install "openai-agents\[litellm\]"

17

It is a common best practice to perform these installations within a virtual environment to manage project dependencies effectively and avoid conflicts.3

### **2.3 Environment Setup (Virtual Environment, API Key)**

1. Create and Activate a Virtual Environment:  
   Before installing the SDK, it is recommended to create and activate a Python virtual environment. This isolates the project's dependencies.  
   Bash  
   python \-m venv env \# or python3 \-m venv env  
   source env/bin/activate \# On Linux/macOS  
   \# env\\Scripts\\activate \# On Windows

   3  
2. Set OpenAI API Key:  
   The SDK requires an OpenAI API key for interacting with OpenAI models and services, including tracing. This key should be set as an environment variable for security and ease of use.  
   Bash  
   export OPENAI\_API\_KEY='sk-...' \# On Linux/macOS  
   \# set OPENAI\_API\_KEY=sk-... \# On Windows

   4  
   Alternatively, the python-dotenv package can be used to manage API keys from a .env file.19 API keys should never be hardcoded directly into the application code, especially for production systems.22

### **2.4 "Hello World": Basic Agent Examples**

Once the SDK is installed and the environment is configured, developers can start building agents.

**2.4.1 Single Agent Example**

This example demonstrates the creation and execution of a single, simple agent.

Python

\# Ensure OPENAI\_API\_KEY is set as an environment variable  
from agents import Agent, Runner

\# Create an agent  
agent \= Agent(name="Assistant", instructions="You are a helpful assistant")

\# Run the agent synchronously  
result \= Runner.run\_sync(agent, "Write a haiku about recursion in programming.")

\# Print the agent's final output  
print(result.final\_output)

3  
The expected output would be a haiku, such as:

Code within the code,  
Functions calling themselves,  
Infinite loop's dance.

3

For Jupyter notebooks or environments with an existing event loop, await Runner.run() can be used.25

**2.4.2 Multi-Agent Handoff Example**

This example illustrates a basic multi-agent scenario where a triage agent hands off a request to a specialized agent based on the language of the input.

Python

import asyncio  
from agents import Agent, Runner

\# Define specialized agents  
spanish\_agent \= Agent(name="Spanish agent", instructions="You only speak Spanish.")  
english\_agent \= Agent(name="English agent", instructions="You only speak English")

\# Define a triage agent with handoff capabilities  
triage\_agent \= Agent(  
    name="Triage agent",  
    instructions="Handoff to the appropriate agent based on the language of the request.",  
    handoffs=\[spanish\_agent, english\_agent\],  
)

async def main\_multi\_agent():  
    \# Run the triage agent with a Spanish input  
    result \= await Runner.run(triage\_agent, input\="Hola, ¿cómo estás?")  
    print(result.final\_output)

if \_\_name\_\_ \== "\_\_main\_\_":  
    \# Patch asyncio for environments like Jupyter that might have a running loop  
    \# import nest\_asyncio  
    \# nest\_asyncio.apply() \# Often needed in Jupyter notebooks  
    asyncio.run(main\_multi\_agent())

3  
The expected output would be a response in Spanish from the spanish\_agent, demonstrating a successful handoff. These introductory examples showcase the fundamental simplicity of creating and running agents with the SDK, aligning with its design goal of being "quick to learn" with "few enough primitives".4 This ease of entry is pivotal for developer adoption, allowing for rapid prototyping and experimentation with agentic workflows.

## **3\. Core Architecture and Components of the OpenAI Agents SDK**

The OpenAI Agents SDK is architected around a small set of powerful primitives designed to offer flexibility and ease of use for building multi-agent applications.3 Understanding these core components—the agent loop, agent definitions, tools, handoffs, guardrails, the runner class, the underlying Responses API, and context management—is essential for effectively leveraging the SDK.

### **3.1 The Agent Loop: Implicit Routines and Orchestration**

At the heart of the SDK's execution model is the **agent loop**. This loop is managed by the Runner class and dictates the iterative process an agent undergoes to fulfill a task.2 When Runner.run() (or its variants) is called, the following sequence typically occurs:

1. The LLM is invoked for the current agent with the current input (conversation history and new user message).  
2. The LLM produces an output.  
   * If the output is a final\_output (i.e., a text response of the desired type with no further tool calls or handoffs), the loop terminates, and the result is returned.  
   * If the LLM initiates a **handoff** to another agent, the current agent is updated to the new agent, the input is prepared for the new agent, and the loop continues from step 1 with the new agent.  
   * If the LLM produces **tool calls**, these tools are executed. Their results are appended to the conversation history, and the loop continues from step 1, feeding the updated history back to the LLM.  
3. The loop continues until a final output is produced or a max\_turns limit (a parameter to prevent infinite loops) is reached, at which point a MaxTurnsExceeded exception is raised.3

The concept of "Routines," explicitly defined in the experimental Swarm framework as predefined sets of steps 6, is handled implicitly within the Agents SDK. Instead of explicit Routine objects, the sequence of operations an agent performs emerges from its instructions, the available tools, potential handoffs, and the LLM's decision-making process within the agent loop.3 This LLM-driven orchestration makes the system more dynamic and adaptable compared to hardcoded sequences. The agent loop, therefore, serves as the engine that abstracts the complexities of these iterative LLM calls, tool integrations, and intermediate state management, simplifying development by allowing a focus on declarative agent definitions.

### **3.2 Agents: Definition, Configuration, and Lifecycle**

The Agent class is the fundamental building block in the SDK, representing an LLM configured with specific instructions, tools, and capabilities for interaction.3

Key Parameters of the Agent Class:  
The Agent class is highly configurable through various parameters defined in its source code 13:

* name (str): A unique identifier for the agent, used in tracing and handoffs.  
* instructions (str | Callable, Agent\], MaybeAwaitable\[str\]\] | None): The system prompt defining the agent's behavior, purpose, personality, and limitations. This can be a static string or a function that dynamically generates instructions based on context.  
* model (str | Model | None): Specifies the LLM to be used (e.g., "gpt-4o-mini"). Defaults to a pre-configured OpenAI model if not set. Allows integration with various models, including non-OpenAI ones via ModelProvider or LiteLLM.18  
* tools (list): A list of Tool objects (e.g., WebSearchTool, custom function tools) that the agent can utilize to perform actions or gather information.  
* handoffs (list\[Agent\[Any\] | Handoff\]): A list of other Agent instances or Handoff objects to which this agent can delegate tasks.  
* output\_type (type\[Any\] | AgentOutputSchemaBase | None): Defines the expected structure of the agent's final output, often using Pydantic models for typed, structured data extraction. If not provided, the output is a string.28  
* handoff\_description (str | None): A human-readable description used when this agent is a target for a handoff, helping the calling LLM decide when to invoke it.  
* model\_settings (ModelSettings): An object to configure model-specific parameters like temperature, top\_p, max\_tokens, etc..32  
* mcp\_servers (list): List of Model Context Protocol (MCP) servers the agent can use.32  
* mcp\_config (MCPConfig): Configuration for MCP servers.32  
* input\_guardrails (list\]): A list of guardrails to validate the agent's input.32  
* output\_guardrails (list\]): A list of guardrails to validate the agent's final output.32  
* hooks (AgentHooks | None): Callbacks for specific lifecycle events of this agent.32  
* tool\_use\_behavior (Literal\["run\_llm\_again", "stop\_on\_first\_tool"\] | StopAtTools | ToolsToFinalOutputFunction): Configures how tool usage is handled (e.g., whether to call the LLM again after a tool result or stop).32

Agent Lifecycle Hooks (AgentHooks):  
The AgentHooks class allows developers to attach custom logic to specific lifecycle events of an individual agent. These are set on the agent.hooks attribute.32 Methods include:

* on\_start(context, agent): Called before this specific agent is invoked.  
* on\_end(context, agent, output): Called when this agent produces a final output.  
* on\_handoff(context, agent, source\_agent): Called when this agent is being handed off to.  
* on\_tool\_start(context, agent, tool): Called before a tool is invoked by this agent.  
* on\_tool\_end(context, agent, tool, result): Called after a tool is invoked by this agent. These hooks provide fine-grained control and observability points during an agent's execution, complementing the global RunHooks.

### **3.3 Tools: Extending Agent Capabilities**

Tools are crucial for enabling agents to interact with the external world, perform actions, and retrieve information beyond the LLM's inherent knowledge. The Agents SDK supports three main types of tools.37

**3.3.1 Hosted Tools (Web Search, File Search, Computer Use)**

OpenAI provides several pre-built "hosted tools" that run on OpenAI's servers alongside the LLMs. These tools offer common functionalities without requiring developers to implement them.37

* **WebSearchTool**: Grants agents access to real-time information from the internet, providing cited search results. This is useful for tasks requiring current knowledge that the LLM was not trained on.37  
* **FileSearchTool**: Enables semantic search over private documents uploaded by the developer to OpenAI-managed vector stores. It supports metadata filtering and custom reranking for accurate retrieval, effectively providing Retrieval-Augmented Generation (RAG) capabilities.37 This tool is configured with vector\_store\_ids to specify which stores to search.37  
* **ComputerTool**: A more experimental tool (available as a research preview for select developers) that allows agents to interact with computer Graphical User Interfaces (GUIs) by simulating mouse and keyboard actions. This can automate tasks in applications without APIs or legacy software.37

These hosted tools significantly lower the barrier for common agent tasks, particularly information retrieval and basic system interaction.

**3.3.2 Function Tools (@function\_tool, schema generation, custom definition)**

The SDK allows developers to convert any Python function into a tool that agents can call. This is a cornerstone of the SDK's extensibility and Python-first design.3

* **Using the @function\_tool decorator**: This is the simplest way to create a function tool. The SDK automatically generates the necessary JSON schema for the tool's parameters based on the Python function's signature (type hints) and its docstring (for descriptions).15  
  * The decorator accepts parameters like name\_override, description\_override, docstring\_style (e.g., 'google', 'sphinx', 'numpy'), use\_docstring\_info (to enable/disable parsing docstrings for descriptions), failure\_error\_function (to customize error messages sent to the LLM), and strict\_mode (for JSON schema validation).42  
  * Schema generation leverages the inspect module for signatures, griffe for docstring parsing, and pydantic for creating the schema.3

Python  
\# Example from \[3, 15, 42\]  
from agents import Agent, Runner, function\_tool  
import asyncio

@function\_tool  
def get\_weather(city: str) \-\> str:  
    """Gets the current weather for a given city.  
    Args:  
        city: The name of the city.  
    """  
    \# In a real scenario, this would call a weather API  
    return f"The weather in {city} is sunny."

weather\_agent \= Agent(  
    name="WeatherAgent",  
    instructions="You provide weather information using the get\_weather tool.",  
    tools=\[get\_weather\]  
)

\# async def run\_weather\_example():  
\#     result \= await Runner.run(weather\_agent, "What's the weather in London?")  
\#     print(result.final\_output)  
\# asyncio.run(run\_weather\_example())  
This automatic schema generation greatly improves developer ergonomics by reducing boilerplate code.38

* **Custom FunctionTool Definition**: For more control or when the decorator is not suitable, a FunctionTool can be instantiated directly. This requires manually providing name (str), description (str), params\_json\_schema (dict), and an async function on\_invoke\_tool(context, args\_json\_string) that handles the tool's execution.37

The ability to easily transform existing Python code into agent-usable tools is a key strength, promoting modularity and reuse.

**3.3.3 Agents as Tools**

An agent can itself be exposed as a tool to another agent using the agent.as\_tool() method.32 This pattern is distinct from handoffs:

* In a handoff, control is fully transferred to the sub-agent, which then continues the conversation or task.  
* When an agent is used as a tool, the calling (manager) agent invokes the sub-agent, receives a result, and then continues its own processing. The sub-agent does not take over the conversation.33

This is particularly useful for implementing hierarchical agent structures, such as a "manager" agent coordinating several "worker" agents, where the manager queries workers for specific information or sub-task completion before synthesizing a final response.30 This enables more complex orchestration patterns where an agent might need to gather inputs from multiple specialized agents.

### **3.4 Handoffs: Inter-Agent Delegation**

Handoffs are a core mechanism for multi-agent collaboration, allowing one agent to delegate a task or transfer control of a conversation to another specialized agent.1 Handoffs are presented to the LLM as callable tools.45

**Defining Handoffs:**

* The handoffs parameter of the Agent class takes a list of Agent objects or Handoff objects.  
* The handoff() factory function is used to create and customize Handoff objects.45

**Key Parameters of the handoff() function:**

* agent (Agent): The target agent to hand off to (mandatory).  
* tool\_name\_override (str | None): Overrides the default handoff tool name (e.g., transfer\_to\_\<agent\_name\>).  
* tool\_description\_override (str | None): Overrides the default description for the handoff tool.  
* on\_handoff (Callable | None): An asynchronous callback function executed when the handoff is invoked. It receives the RunContextWrapper and optionally LLM-generated input data (if input\_type is specified). Useful for logging or pre-handoff data fetching.  
* input\_type (type | None): A Pydantic model defining the structure of data the LLM should provide when initiating the handoff (e.g., a reason for escalation).  
* input\_filter (Callable, HandoffInputData\] | None): A function to modify the conversation history passed to the receiving agent. This allows for selective context sharing, for example, by removing previous tool calls using predefined filters like agents.extensions.handoff\_filters.remove\_all\_tools.45

To ensure LLMs correctly utilize handoffs, it's recommended to use agents.extensions.handoff\_prompt.prompt\_with\_handoff\_instructions() or include RECOMMENDED\_PROMPT\_PREFIX in agent instructions.41

Python

\# Conceptual example based on \[45\]  
from agents import Agent, handoff, RunContextWrapper  
from agents.extensions.handoff\_prompt import prompt\_with\_handoff\_instructions  
from pydantic import BaseModel  
import asyncio

class EscalationDetails(BaseModel):  
    reason: str  
    urgency: int

async def log\_escalation(ctx: RunContextWrapper\[None\], input\_data: EscalationDetails):  
    print(f"Handoff to Billing Specialist. Reason: {input\_data.reason}, Urgency: {input\_data.urgency}")

billing\_specialist\_agent \= Agent(  
    name="BillingSpecialistAgent",  
    instructions="You are a billing specialist. Handle complex billing inquiries."  
)

customer\_service\_agent \= Agent(  
    name="CustomerServiceAgent",  
    instructions=prompt\_with\_handoff\_instructions(  
        "You are a customer service agent. Handle general inquiries. "  
        "If a billing issue is complex and requires specialist knowledge, "  
        "use the 'escalate\_to\_billing\_specialist' tool, providing a reason and urgency level."  
    ),  
    handoffs=  
)  
\# Example usage:  
\# async def run\_handoff\_example():  
\#     result \= await Runner.run(  
\#         customer\_service\_agent,  
\#         "My bill is incorrect and very complicated, I need an expert."  
\#     )  
\#     print(result.final\_output)  
\# asyncio.run(run\_handoff\_example())

The handoff mechanism, with its customization options, provides a structured way to manage complex, multi-stage tasks by distributing responsibilities among specialized agents.

### **3.5 Guardrails: Ensuring Safe and Validated Interactions**

Guardrails are a critical feature for building safe and reliable agentic applications. They run in parallel to agent execution, performing validation checks on inputs and outputs.1

* **Input Guardrails**: Execute on the initial user input. They only run if the agent is the first in an execution chain.49  
* **Output Guardrails**: Execute on the final output of an agent. They only run if the agent produces the final output of the execution chain.49

Guardrails are defined as functions, typically decorated with @input\_guardrail or @output\_guardrail, or by creating InputGuardrail or OutputGuardrail class instances.49 These functions receive the agent's input (or output) and context, and must return a GuardrailFunctionOutput object.

The GuardrailFunctionOutput contains two key fields 13:

* output\_info (Any): Optional information about the guardrail's checks.  
* tripwire\_triggered (bool): If True, it signifies that the input or output failed the guardrail check.

If tripwire\_triggered is True, the SDK immediately raises an InputGuardrailTripwireTriggered or OutputGuardrailTripwireTriggered exception, halting further agent execution.49 This is crucial for preventing misuse (e.g., an expensive model being used for off-topic requests) or the propagation of undesirable outputs.

Python

\# Conceptual example based on \[49\]  
from agents import Agent, GuardrailFunctionOutput, input\_guardrail, Runner, InputGuardrailTripwireTriggered  
from pydantic import BaseModel  
import asyncio

class ProfanityCheckOutput(BaseModel):  
    contains\_profanity: bool  
    details: str

profanity\_checker\_agent \= Agent(  
    name="ProfanityCheckerAgent",  
    instructions="Check if the input text contains profanity. Respond with structured output.",  
    output\_type=ProfanityCheckOutput  
)

@input\_guardrail  
async def profanity\_input\_guardrail(ctx, agent, input\_data) \-\> GuardrailFunctionOutput:  
    \# In a real scenario, input\_data might be a list of TResponseInputItem  
    \# For simplicity, assuming input\_data is a string query here.  
    if isinstance(input\_data, list): \# Basic handling for list input  
        user\_message \= next((item.get("content") for item in input\_data if item.get("role") \== "user"), "")  
    else:  
        user\_message \= input\_data

    result \= await Runner.run(profanity\_checker\_agent, user\_message, context=ctx.context)  
    final\_output \= result.final\_output\_as(ProfanityCheckOutput)  
    return GuardrailFunctionOutput(  
        output\_info=final\_output,  
        tripwire\_triggered=final\_output.contains\_profanity  
    )

main\_task\_agent \= Agent(  
    name="MainTaskAgent",  
    instructions="You are a helpful assistant for general queries.",  
    input\_guardrails=\[profanity\_input\_guardrail\]  
)

\# async def run\_guardrail\_example():  
\#     try:  
\#         await Runner.run(main\_task\_agent, "Tell me a darn joke.") \# Assuming 'darn' trips the profanity checker  
\#     except InputGuardrailTripwireTriggered:  
\#         print("Input Guardrail Tripped: Profanity detected.")  
\#     else:  
\#         print("Query processed successfully.")  
\# asyncio.run(run\_guardrail\_example())

This proactive validation mechanism is essential for building responsible and secure AI systems. The design choice to co-locate guardrail definitions with the agent (on the Agent class) enhances readability and maintainability, as guardrails are often specific to an agent's function.49

### **3.6 Runner Class: Executing Agent Workflows**

The Runner class is responsible for executing agent workflows.3 It provides three primary methods for execution:

* Runner.run(starting\_agent, input, \*, context=None, max\_turns=DEFAULT\_MAX\_TURNS, hooks=None, run\_config=None, previous\_response\_id=None): An asynchronous method that executes the agent loop and returns a RunResult object upon completion.52  
* Runner.run\_sync(...): A synchronous wrapper around Runner.run(), useful for simpler scripts or environments where async execution is not required.3  
* Runner.run\_streamed(...): An asynchronous method that executes the agent loop and returns a RunResultStreaming object. This method calls the LLM in streaming mode and allows the developer to process events (like token generation) as they occur, which is ideal for real-time user interfaces.3

The input parameter can be a string (representing a user message) or a list of TResponseInputItem objects (conforming to the OpenAI Responses API structure).29

The run\_config parameter (of type RunConfig) allows for global settings for the entire agent run, such as 29:

* model: A global LLM model to use, overriding individual agent settings.  
* model\_provider: A provider for looking up model names.  
* model\_settings: Global model tuning parameters (e.g., temperature).  
* input\_guardrails, output\_guardrails: Global guardrails applied to all runs.  
* handoff\_input\_filter: A global filter for all handoffs.  
* tracing\_disabled (bool): Disables tracing for the run.  
* trace\_include\_sensitive\_data (bool): Configures whether traces include potentially sensitive data.  
* workflow\_name, trace\_id, group\_id: For tracing and grouping runs.  
* trace\_metadata: Custom metadata for traces.

The Runner class thus provides a unified and flexible interface for initiating and managing the execution of agents, catering to different application requirements from batch processing to interactive streaming.

### **3.7 Responses API: The Foundation for Agent Interactions**

The OpenAI Agents SDK leverages the **Responses API** as its primary interface for interacting with OpenAI models, especially when built-in tools are involved.18 This API is designed to be a more flexible and powerful successor to the Chat Completions API for agentic applications, combining the simplicity of Chat Completions with the advanced tool-use capabilities previously associated with the Assistants API.39

Key characteristics of the Responses API relevant to the Agents SDK:

* **Designed for Tool Use**: It natively supports complex, multi-turn interactions involving multiple tool calls by the LLM.38  
* **Built-in Tools**: It enables the use of OpenAI's hosted tools like web search, file search, and computer use directly within the API call flow.39  
* **Unified Interface**: It aims to simplify development by providing a single API endpoint for complex agent behaviors that might have previously required juggling multiple API calls or types.39  
* **Recommended Model Class**: Within the Agents SDK, the OpenAIResponsesModel class is the recommended way to interact with OpenAI models using this API.18

While the Chat Completions API remains supported (and can be used via OpenAIChatCompletionsModel in the SDK), particularly for scenarios not requiring built-in tools 18, the Responses API is positioned as the future direction for building agents on OpenAI's platform.39 This strategic direction underscores its importance as the foundational communication layer through which agents in the SDK interact with OpenAI's most advanced models and their integrated capabilities.

### **3.8 Context Management and State Across Turns**

Effective context management is crucial for coherent and intelligent agent behavior across multiple interaction turns. The OpenAI Agents SDK distinguishes between two main types of context 53:

1. **Local Context (Application State/Dependencies)**:  
   * This refers to data and dependencies that are available to the Python code during an agent run, such as in tool functions, handoff callbacks, or lifecycle hooks.  
   * It is managed by passing a Python object (often a dataclass or Pydantic model instance) to the context parameter of the Runner.run() methods.  
   * Inside tool functions or hooks, this context object can be accessed via RunContextWrapper.context, where TContext is the type of the passed object.53  
   * This local context is *not* sent to the LLM; it's purely for the local Python execution environment. It can be used to hold user session information, database connections, logger objects, or helper functions.16

Python  
\# Example from \[53\]  
import asyncio  
from dataclasses import dataclass  
from agents import Agent, RunContextWrapper, Runner, function\_tool

@dataclass  
class UserInfo:  
    name: str  
    uid: int

@function\_tool  
async def fetch\_user\_age(wrapper: RunContextWrapper\[UserInfo\]) \-\> str:  
    \# Access local context: wrapper.context.name, wrapper.context.uid  
    \# Simulate fetching age based on uid  
    if wrapper.context.uid \== 123:  
        return f"User {wrapper.context.name} is 47 years old"  
    return "User age not found."

\# async def run\_context\_example():  
\#     user\_info\_context \= UserInfo(name="John", uid=123)  
\#     context\_agent \= Agent\[UserInfo\]( \# Specify context type for the agent  
\#         name="ContextAwareAssistant",  
\#         tools=\[fetch\_user\_age\],  
\#         instructions="Use tools to answer questions about the user."  
\#     )  
\#     result \= await Runner.run(  
\#         starting\_agent=context\_agent,  
\#         input="What is the age of the user?",  
\#         context=user\_info\_context,  
\#     )  
\#     print(result.final\_output)  
\# asyncio.run(run\_context\_example())

2. **Agent/LLM Context (Conversation History)**:  
   * This is the information the LLM sees when generating a response, primarily consisting of the conversation history (user messages, assistant messages, tool calls, and tool responses).  
   * To make new data available to the LLM, it must be incorporated into this history.53  
   * Strategies include:  
     * Adding information to the Agent.instructions (system prompt), which can be static or dynamically generated based on the local context.53  
     * Including messages in the input when calling Runner.run().53  
     * Exposing data via function tools, allowing the LLM to request specific information on demand.53  
   * For multi-turn conversations, the state (conversation history) is typically maintained by taking the output of one Runner.run() call and using it to construct the input for the next turn. The RunResultBase.to\_input\_list() method can be used to get the list of all messages from a run, suitable for feeding into the subsequent run.29 This ensures the LLM has the necessary preceding dialogue to maintain coherence.

This clear separation allows developers to manage complex application-specific state locally while ensuring the LLM receives a clean and relevant conversational history for its decision-making processes.

### **3.9 Table 3.1: Core API Components of OpenAI Agents SDK**

| Component | Key Parameters/Methods | Purpose |
| :---- | :---- | :---- |
| Agent | name, instructions, model, tools, handoffs, output\_type | Defines an AI agent's behavior, capabilities, and connections. |
| Runner | run(), run\_sync(), run\_streamed(), RunConfig | Executes agent workflows and manages the agent loop. |
| @function\_tool / Tool | func (for decorator), name\_override, description\_override / WebSearchTool, FileSearchTool | Enables agents to interact with external systems or perform specific actions. |
| handoff() / Handoff | agent (target), on\_handoff (callback), input\_type, input\_filter | Facilitates delegation of tasks and control flow between different agents. |
| @input\_guardrail / @output\_guardrail / GuardrailFunctionOutput | guardrail\_function, tripwire\_triggered | Provides safety checks and validation for agent inputs and outputs. |
| OpenAIResponsesModel | model (name), openai\_client | Recommended interface for OpenAI models, leveraging the Responses API for tool use. |
| RunContextWrapper | context | Provides access to local application context within tools and lifecycle hooks. |
| AgentHooks / RunHooks | on\_agent\_start, on\_tool\_end, etc. | Allow custom logic injection at various lifecycle events of an agent or an entire run. |

This table summarizes the primary building blocks provided by the SDK, offering a high-level view of its architecture.

## **4\. Advanced Features and Capabilities**

Beyond its core components, the OpenAI Agents SDK offers several advanced features that enhance its power and flexibility for building sophisticated multi-agent systems. These include multi-model orchestration, strategies for persistent memory, comprehensive tracing, global lifecycle hooks, and a design focused on developer ergonomics.

### **4.1 Multi-Model Orchestration**

A significant strength of the OpenAI Agents SDK is its provider-agnostic design, allowing developers to orchestrate agents powered by a diverse range of LLMs, not limited to OpenAI's offerings.3

* **Native OpenAI Model Support**: The SDK provides out-of-the-box support for OpenAI models through two main classes:  
  * OpenAIResponsesModel: The recommended class, utilizing the newer Responses API.18  
  * OpenAIChatCompletionsModel: Uses the standard Chat Completions API.18  
* **LiteLLM Integration**: For broader model compatibility, the SDK integrates seamlessly with LiteLLM, a library that provides a unified interface to over 100 LLMs from various providers (e.g., Anthropic, Google, Cohere, local models via Ollama).3 To use a LiteLLM-supported model, developers typically prefix the model name with litellm/ (e.g., litellm/claude-3-haiku-20240307) when configuring an agent.17 This requires installing the openai-agents\[litellm\] dependency group.  
* **Custom Model Configuration**: The SDK offers granular control for specifying models:  
  * **Per Agent**: The model parameter of the Agent class can be set to a model name string (for OpenAI models or LiteLLM-prefixed models) or a Model instance (e.g., OpenAIChatCompletionsModel configured with a custom base\_url for providers like Gemini, or an instance of LitellmModel).17 This allows different agents within the same workflow to use different LLMs.  
  * **Per Run**: The RunConfig object passed to Runner.run() can specify a global model or model\_provider for all agents in that specific run.18  
  * **Global Default**: set\_default\_openai\_client can be used to configure an AsyncOpenAI client globally, useful for OpenAI-compatible API endpoints from other providers.18

Python

\# Conceptual Example for LiteLLM and Per-Agent Model Specification  
\# (Assumes OPENAI\_API\_KEY is set for tracing, and relevant keys for other models are in env for LiteLLM)  
\# pip install "openai-agents\[litellm\]"  
from agents import Agent, Runner, ModelSettings  
from agents.extensions.models.litellm\_model import LitellmModel \# Actual import path may vary  
from openai import AsyncOpenAI  
from agents.models import OpenAIChatCompletionsModel  
import asyncio  
import os

\# Example: Using Anthropic Claude via LiteLLM  
claude\_model\_identifier \= "litellm/claude-3-opus-20240229" \# Ensure this is a valid LiteLLM model string  
\# Ensure ANTHROPIC\_API\_KEY is set in environment for LiteLLM  
claude\_agent \= Agent(  
    name="ClaudeAgent",  
    instructions="You are an assistant powered by Anthropic's Claude.",  
    model=LitellmModel(model=claude\_model\_identifier) \# Or simply model=claude\_model\_identifier if LitellmModel is default for litellm/ prefix  
)

\# Example: Using a local model via Ollama and LiteLLM  
\# Ensure Ollama is running and has the 'mistral' model  
ollama\_mistral\_agent \= Agent(  
    name="OllamaMistralAgent",  
    instructions="You are a helpful assistant running locally via Ollama and LiteLLM.",  
    model=LitellmModel(model="litellm/ollama/mistral") \# Or "litellm/mistral" if Ollama is default local handler  
)

\# Example: Using OpenAI's gpt-4o-mini for a specific agent  
openai\_fast\_agent \= Agent(  
    name="OpenAIFastAgent",  
    instructions="You are a quick and efficient assistant.",  
    model="gpt-4o-mini",  
    model\_settings=ModelSettings(temperature=0.5)  
)

\# Triage agent that could hand off to any of the above  
triage\_agent \= Agent(  
    name="TriageMaster",  
    instructions="Determine the best specialized agent for the query.",  
    handoffs=\[claude\_agent, ollama\_mistral\_agent, openai\_fast\_agent\],  
    model="gpt-3.5-turbo" \# Triage agent itself uses a different model  
)

\# async def run\_multi\_model\_example():  
\#     result\_claude \= await Runner.run(claude\_agent, "Explain the concept of emergence in complex systems.")  
\#     print(f"Claude Agent Output: {result\_claude.final\_output}")

\#     result\_ollama \= await Runner.run(ollama\_mistral\_agent, "What is the capital of France?")  
\#     print(f"Ollama Mistral Agent Output: {result\_ollama.final\_output}")

\#     result\_triage \= await Runner.run(triage\_agent, "I need a very detailed explanation of quantum entanglement.")  
\#     print(f"Triage Agent routed, final output: {result\_triage.final\_output}") \# This would come from one of the specialist agents

\# asyncio.run(run\_multi\_model\_example())

This flexibility allows developers to optimize for cost, performance, or specific model capabilities on a per-agent basis, which is crucial for building efficient and effective multi-agent systems. However, developers should be mindful of potential inconsistencies in feature support (e.g., structured outputs, hosted tools) when mixing models from different providers.18

### **4.2 Persistent Memory Strategies**

The OpenAI Agents SDK itself does not provide built-in mechanisms for persistent memory that lasts across separate agent runs or conversations.4 The SDK primarily focuses on the orchestration and execution logic for single or multi-agent interactions within a given run. Conversation history within a single, continuous interaction (across multiple turns) is managed by passing the outputs of one turn as inputs to the next, often facilitated by methods like RunResultBase.to\_input\_list().29

For long-term memory or persistence across distinct sessions, developers are responsible for implementing their own solutions. This is typically achieved by:

* **Using Custom Tools**: Creating function tools that interact with external storage systems such as vector databases (e.g., Pinecone, ChromaDB, Faiss), graph databases (e.g., FalkorDB 54), key-value stores, or traditional SQL/NoSQL databases. These tools can save and retrieve agent state, conversation summaries, user preferences, or relevant knowledge.  
* **OpenAI's FileSearchTool**: This hosted tool allows agents to retrieve information from documents stored in OpenAI-managed vector stores, providing a form of RAG-based persistent knowledge.37 While OpenAI manages the storage, the developer controls the content and its use via the tool.  
* **Third-Party Memory Services**: Integrating dedicated memory services like Zep 55 or Mem0 56 through custom tools. These services often provide more sophisticated memory functionalities like semantic search over past conversations, summarization, and entity extraction.  
  * The Mem0 example demonstrates creating save\_memories and search\_memories function tools to interact with the Mem0 client.56  
* **LangChain's LangMem SDK**: While a separate library, LangMem by LangChain offers patterns for semantic (facts, knowledge), episodic (past experiences, few-shot examples), and procedural (evolving behavior via prompt updates) memory, which could conceptually be integrated into Agents SDK via tools.57

The context object passed to Runner.run() is for managing local, in-run state and dependencies, not for inter-run persistence.16 While the SDK is stateless regarding long-term memory, its tool-based architecture provides the necessary extensibility for developers to integrate any memory solution they choose, offering flexibility at the cost of requiring explicit implementation of persistence logic.

### **4.3 Built-in Tracing and Observability**

The OpenAI Agents SDK includes a robust, built-in tracing system designed to provide observability into agent execution flows.1

* **Automatic Tracing**: By default, agent runs are automatically traced. These traces capture detailed information about the workflow, including LLM calls, tool invocations and their results, handoffs between agents, and guardrail executions.  
* **OpenAI Platform Integration**: Traces can be viewed on the OpenAI platform's dashboard ([platform.openai.com/traces](https://platform.openai.com/traces)), allowing developers to visualize, debug, and monitor their agentic workflows.14 This is particularly useful during development and for production monitoring.  
* **Configuration**: Tracing behavior can be configured via the RunConfig object passed to Runner.run(). Options include:  
  * tracing\_disabled: To turn off tracing.  
  * trace\_include\_sensitive\_data: To control whether potentially sensitive data (LLM inputs/outputs, tool call data) is included in traces.  
  * workflow\_name, trace\_id, group\_id: To organize and identify traces.29  
* **Extensibility**: The tracing system is designed to be extensible. It supports custom spans and can export trace data to a wide variety of external observability platforms, including Logfire, AgentOps, Braintrust, Scorecard, and Keywords AI.3 Other platforms like MLflow 62, Langfuse 63, and Arize Phoenix 65 also provide integrations for tracing OpenAI calls, which can be adapted for Agents SDK workflows.

This built-in and extensible tracing capability is crucial for managing the complexity of multi-agent systems, enabling developers to understand agent behavior, identify bottlenecks, and optimize performance effectively.

### **4.4 Lifecycle Hooks (RunHooks)**

In addition to AgentHooks which are specific to individual agents (Section 3.2), the SDK provides RunHooks for global lifecycle event interception during an agent run.36 These hooks are passed to the Runner.run(hooks=...) method.52

The RunHooks class is a generic class that developers can subclass to implement callbacks for various events occurring across the entire execution flow, regardless of which agent is currently active. Key RunHooks methods include:

* async on\_agent\_start(context: RunContextWrapper, agent: Agent): Called before any agent in the run is invoked, and each time the current agent changes.  
* async on\_agent\_end(context: RunContextWrapper, agent: Agent, output: Any): Called when an agent (any agent in the run) produces a final output.  
* async on\_handoff(context: RunContextWrapper, from\_agent: Agent, to\_agent: Agent): Called when a handoff occurs between any two agents.  
* async on\_tool\_start(context: RunContextWrapper, agent: Agent, tool: Tool): Called before any tool is invoked by any agent.  
* async on\_tool\_end(context: RunContextWrapper, agent: Agent, tool: Tool, result: str): Called after any tool has been invoked.

RunHooks provide a powerful mechanism for implementing cross-cutting concerns such as centralized logging, metrics collection for the entire run, resource management (setup/teardown for the run), or dynamic modifications to the execution flow at a global level.

### **4.5 Developer Ergonomics and Python-First Design**

A core design philosophy of the OpenAI Agents SDK is to prioritize **developer ergonomics** and maintain a **Python-first approach**.3

* **Minimal Abstractions**: The SDK is intentionally lightweight, introducing very few new abstractions. This allows developers to leverage their existing Python knowledge and built-in language features for orchestrating and chaining agents, rather than needing to learn a complex new framework-specific syntax or paradigm.1  
* **Ease of Use**: The aim is to make the SDK quick to learn and use, enabling rapid prototyping and development of agentic applications.4  
* **Automatic Schema Generation**: A key feature enhancing developer experience is the automatic generation of JSON schemas for function tools from Python type hints and docstrings.37 This significantly reduces boilerplate code and the potential for errors associated with manually defining tool interfaces for LLMs.  
* **Flexibility and Customization**: While designed to work well "out of the box," the SDK also allows developers to customize "exactly what happens" 4, providing control when needed.

This focus on simplicity and leveraging native Python constructs aims to make the SDK accessible to a broad range of Python developers, lowering the barrier to entry for building sophisticated multi-agent systems.

## **5\. Orchestrating Multi-Agent Systems: Patterns and Best Practices**

The OpenAI Agents SDK provides flexible primitives that support various approaches to orchestrating interactions among multiple agents. Effective orchestration is key to building robust and intelligent multi-agent systems. This section explores different orchestration paradigms, best practices for designing agent instructions, common architectural patterns, and strategies for error handling and debugging.

### **5.1 LLM-Driven Orchestration vs. Code-Driven Orchestration**

There are two primary paradigms for orchestrating agent workflows using the SDK 30:

1. **LLM-Driven Orchestration**:  
   * In this approach, the LLM itself is responsible for planning, reasoning, and deciding the sequence of actions. Based on its instructions and the available tools and handoffs, the agent autonomously determines whether to call a tool, hand off to another agent, or respond directly to the user.31  
   * This paradigm is well-suited for open-ended tasks where the path to a solution is not predefined and requires dynamic adaptation based on unfolding context. For example, a research agent might dynamically decide to use web search, then file search, then hand off to a report-writing agent based on its findings.31  
   * The effectiveness of LLM-driven orchestration heavily relies on the quality of agent instructions, the clarity of tool descriptions, and the LLM's reasoning capabilities.  
2. **Code-Driven Orchestration**:  
   * Here, the flow of agents and the logic for transitions are explicitly defined in the developer's Python code, rather than being left to the LLM's discretion.31  
   * This approach offers greater determinism, predictability, and control over the workflow, which can be beneficial for optimizing speed, cost, and performance, especially for well-defined tasks.  
   * Common patterns include:  
     * Using structured outputs from one agent (e.g., a classification) to programmatically decide which agent to invoke next.  
     * Chaining agents sequentially, where the output of one agent becomes the input for the next (e.g., a blog post generation pipeline: research \-\> outline \-\> draft \-\> critique \-\> revise).  
     * Implementing loops where an agent performs a task, and an evaluator agent provides feedback, with the loop continuing until certain criteria are met.31

The Agents SDK is flexible enough to support both paradigms. Developers can choose to give more autonomy to the LLM for complex, dynamic tasks or implement more structured, code-driven flows for predictable processes. Often, a hybrid approach, where some parts of the workflow are LLM-driven and others are code-driven, can provide the optimal balance of flexibility and reliability.

### **5.2 Designing Effective Agent Instructions**

The quality of instructions provided to an agent is paramount, as these instructions (system prompts) are the primary means of guiding the LLM's behavior, decision-making, and interaction style.19 Clear, specific, and comprehensive instructions reduce ambiguity and improve the reliability of agent actions.

Best practices for designing agent instructions include 30:

* **Clarity and Specificity**: Clearly define the agent's role, purpose, personality (if any), and limitations. Be explicit about what the agent should and should not do.  
* **Leverage Existing Documentation**: For tasks like customer service, existing support scripts, policy documents, or knowledge base articles can be transformed into LLM-friendly instructions or routines.30  
* **Task Decomposition**: Prompt agents to break down complex tasks into smaller, manageable steps. This helps the LLM follow instructions more effectively and reduces the likelihood of errors.30  
* **Define Clear Actions**: Each step in an agent's intended routine should correspond to a specific action (e.g., "ask the user for their order ID," "call the get\_order\_status tool") or a type of output. Explicitly stating actions and even desired phrasing for user-facing messages minimizes misinterpretation.30  
* **Tool Usage Guidance**: Clearly describe available tools, their parameters, and when and how they should be used. Good tool descriptions are crucial for the LLM to make correct tool selections.  
* **Handoff Guidance**: If handoffs are involved, provide clear instructions on when to hand off and to which agent, often aided by handoff\_description and functions like prompt\_with\_handoff\_instructions.41  
* **Handle Edge Cases**: Anticipate common variations, potential errors, or incomplete information from users, and include instructions on how to handle these situations (e.g., conditional steps, requests for clarification).30  
* **Iterative Refinement**: Monitor agent behavior and iteratively refine instructions based on observed performance and failures. Evaluation frameworks and tracing are invaluable for this process.31  
* **Specialization**: Favor specialized agents that excel at a narrow set of tasks over general-purpose agents expected to handle everything, as this generally leads to better performance and simpler instructions.31  
* **Automated Instruction Generation**: For converting existing documents into agent instructions, advanced models like o1 or o3-mini can be prompted to perform this transformation, ensuring clarity and an LLM-friendly format.30

Effective instruction design is an ongoing process of prompting, testing, and refinement, crucial for harnessing the full potential of LLM-driven agents.

### **5.3 Common Agent Patterns (Triage, Manager-Worker, Sequential Pipelines)**

The OpenAI Agents SDK's primitives (agents, tools, handoffs) enable the implementation of various established multi-agent system (MAS) architectural patterns. The choice of pattern depends on the specific problem domain and desired interaction dynamics. Some common patterns found in examples and documentation include:

* **Triage/Routing**:  
  * **Description**: A central triage agent receives an initial request and, based on its content or user intent, routes it to one of several specialized agents. This is a common pattern in customer service or information systems.  
  * **SDK Implementation**: Typically implemented using an Agent with multiple handoffs defined. The triage agent's instructions guide it to select the appropriate handoff tool.3  
  * **Example**: A customer support system where a triage agent hands off to BillingAgent, TechnicalSupportAgent, or FAQAgent.30 The examples/agent\_patterns/routing.py in the SDK repository demonstrates this.26  
* **Manager-Worker (Agents as Tools)**:  
  * **Description**: A "manager" or "orchestrator" agent coordinates the work of several "worker" or "specialist" agents. The manager agent calls the worker agents as tools, gathers their outputs, and may synthesize a final result or make further decisions. The manager retains control throughout the process.  
  * **SDK Implementation**: The worker agents are exposed as tools to the manager agent using the agent.as\_tool() method.30  
  * **Example**: A research manager agent that calls a WebSearchAgentTool, a DataAnalysisAgentTool, and a ReportWriterAgentTool to produce a comprehensive report.  
* **Sequential Pipelines (Chaining Agents)**:  
  * **Description**: Tasks are broken down into a sequence of steps, with each step handled by a dedicated agent. The output of one agent becomes the input for the next agent in the chain. This is suitable for well-defined, multi-stage processes.  
  * **SDK Implementation**: This can be achieved through code-driven orchestration, where the Runner.run() method is called sequentially for each agent, passing the relevant data between them. Handoffs can also model simpler sequential flows if the LLM is instructed appropriately.31  
  * **Example**: A content creation pipeline: TopicResearchAgent \-\> OutlineGeneratorAgent \-\> DraftWriterAgent \-\> ReviewerAgent \-\> EditorAgent.  
* **Parallel Execution**:  
  * **Description**: Multiple agents or tasks that do not depend on each other are executed concurrently to improve speed and efficiency.  
  * **SDK Implementation**: Python's asyncio.gather can be used to run multiple Runner.run() calls for different agents in parallel, with their results collected afterwards.31  
  * **Example**: Fetching information from multiple independent sources simultaneously, with each source handled by a separate agent.  
* **Iterative Refinement (Agent-Critique Loop)**:  
  * **Description**: One agent performs a task (e.g., writing text), and another "evaluator" or "critique" agent reviews the output and provides feedback. The first agent then revises its work based on the feedback, and this loop continues until the output meets certain quality criteria or a maximum number of iterations is reached.  
  * **SDK Implementation**: This involves code-driven orchestration of a loop containing Runner.run() calls for the task agent and the critique agent.31  
  * **Example**: An agent writing a summary, with another agent checking for factual accuracy and clarity, iterating until the summary is satisfactory.

The examples/agent\_patterns directory in the openai-agents-python GitHub repository provides practical implementations of some of these patterns.3 The SDK's flexibility allows developers to combine these patterns or create custom orchestration logic as needed.

### **5.4 Error Handling and Debugging Strategies**

Robust error handling and effective debugging are essential for developing reliable agentic applications, especially given the non-deterministic nature of LLMs and the complexity of multi-agent interactions. The OpenAI Agents SDK provides several mechanisms and best practices:

* **Tracing for Debugging**:  
  * The built-in tracing feature is the primary tool for debugging. Traces capture detailed information about each step in an agent run, including LLM inputs/outputs, tool calls (parameters and results), handoffs, and guardrail evaluations.3  
  * When an agent fails or behaves unexpectedly, reviewing the trace on the OpenAI platform or an integrated observability tool can help identify the point of failure, such as incorrect tool parameters, misrouted handoffs, or API errors from tools or LLMs.9  
* **Tool Error Handling**:  
  * When using the @function\_tool decorator, the failure\_error\_function parameter can be used to customize how errors occurring within the tool are reported back to the LLM.37  
    * The default behavior is to use default\_tool\_error\_function, which informs the LLM that an error occurred.  
    * Providing a custom function allows for more specific error messages to be sent to the LLM, potentially enabling it to self-correct or retry.  
    * If failure\_error\_function is set to None, exceptions raised within the tool will propagate up, potentially halting the agent run if not caught by the developer's code. This is useful if external error handling logic is preferred.  
  * It's good practice for tools to return clear, informative error messages (e.g., "Invalid Order ID provided") rather than generic exceptions, to help the LLM understand the issue.9  
* **Handoff Error Handling**:  
  * For critical handoffs, consider including a fallback agent or error handling logic in the on\_handoff callback or in the orchestrating code to manage situations where a handoff fails or the receiving agent encounters an issue.9  
* **Run-Level Exceptions**:  
  * MaxTurnsExceeded: Raised by Runner.run() if the agent loop executes more than the max\_turns limit, preventing infinite loops.29  
  * InputGuardrailTripwireTriggered / OutputGuardrailTripwireTriggered: Raised if an input or output guardrail's tripwire\_triggered flag is true, halting execution.49 These exceptions should be caught and handled appropriately (e.g., by informing the user or logging the incident).  
* **General Best Practices for Workflow Design** 9:  
  * **Modularity**: Design agents to be modular, with each handling specific, well-defined tasks. This simplifies debugging and error isolation.  
  * **Clarity in Instructions**: Write unambiguous instructions for agents to minimize unexpected behavior.  
  * **Validation**: Apply guardrails at key stages of the workflow to catch issues early.  
  * **Idempotency**: Design tools to be idempotent where possible, so retrying a failed tool call does not have unintended side effects.  
  * **Logging**: Implement comprehensive logging within tool functions, lifecycle hooks, and orchestration code to supplement the SDK's tracing.

By combining the SDK's built-in tracing with careful error handling in custom tools and orchestration logic, developers can build more resilient and debuggable multi-agent systems.

## **6\. Use Cases and Practical Applications (Jan 2023 – May 26 2025\)**

The OpenAI Agents SDK, and the conceptual groundwork laid by its precursor "Swarm," have spurred development and experimentation across a wide array of use cases. The ability to orchestrate multiple specialized agents, equip them with tools, and manage their interactions via handoffs has proven valuable in domains requiring complex task decomposition, information retrieval, and decision-making. The period between January 2023 and May 26, 2025, has seen a growing exploration of these capabilities.

**Key Application Domains:**

* **Customer Support and Service**: This is a prominent area where multi-agent systems excel.  
  * **Triage Systems**: An initial agent analyzes customer queries and routes them to specialized agents for sales, refunds, technical support, or FAQs.2 The TriageAgent example in the SDK documentation, handing off to AccountAgent, KnowledgeAgent, and SearchAgent, is a prime illustration.41  
  * **Automated Dispute Management**: Systems can automate the handling of payment disputes by integrating with APIs like Stripe. A triage agent can decide to accept or escalate a dispute, an acceptance agent can process clear-cut cases, and an investigator agent can gather evidence for complex disputes.27  
  * **Airline Customer Service**: Example systems demonstrate handling flight bookings, cancellations, and inquiries.68  
* **Research and Information Synthesis**:  
  * **Deep Research Assistants**: Agents can be equipped with web search tools (WebSearchTool) and file search tools (FileSearchTool for RAG over private documents) to gather, analyze, and synthesize information from diverse sources.1 The research\_bot example in the SDK repository mimics deep research capabilities.68  
  * **News Collection**: Agents can be designed to monitor and collect news articles based on specific criteria.71  
* **E-commerce and Retail**:  
  * Agents can assist with product recommendations, process returns, and provide a more interactive shopping experience.68  
* **Finance and Banking**:  
  * Applications include automated fraud detection, personalized financial advisory, and customer support for banking inquiries.72 The dispute management use case with Stripe also falls into this category.27  
* **Healthcare**:  
  * Agents can assist in patient data collection, triage based on symptoms, and provide preliminary medical recommendations (under appropriate supervision and ethical guidelines).70  
* **Robotics and Automation**:  
  * While more conceptual for the SDK itself, the principles of multi-agent orchestration are directly applicable to coordinating robots in an assembly line or managing autonomous vehicles, where each agent handles specialized sensor inputs or control tasks.70  
* **Software Development and DevOps**:  
  * **Codebase Interaction**: Agents can assist with understanding codebases, writing features, fixing bugs, and proposing pull requests.74  
  * **GitHub Repository Management**: Agents can be built to create issues, update repository metadata, and assist with pull requests, as demonstrated by examples using Composio to connect agents to GitHub.20  
  * **Threat Modeling for AI Agents**: Tools are being developed to use agents for threat modeling and visualizing AI agent systems built with frameworks like the Agents SDK.71  
  * **CI/CD Pipeline Planning**: Agents can scan repositories and propose cloud architecture and CI/CD pipeline plans, including writing Terraform code and GitHub Actions files.75  
* **Content Generation and Management**:  
  * **Social Media Content**: Systems can generate engaging social media posts using specialized agents for trend finding, content writing, and visual concept suggestion.71  
  * **Notion Page Management**: Agents can create and manage Notion pages, add tasks, and automate status updates.20  
* **Personal Productivity**:  
  * Automated personal assistants can manage scheduling, reminders, and email responses.70  
* **Travel Planning**:  
  * AI-powered travel planning systems can optimize budgets, search multiple sources for flights and accommodations, and provide personalized recommendations.15

**Illustrative Code Snippets for Key Use Cases:**

1. **Customer Service Triage (Conceptual based on** 41**)**:  
   Python  
   from agents import Agent, Runner, WebSearchTool, FileSearchTool, function\_tool  
   from agents.extensions.handoff\_prompt import prompt\_with\_handoff\_instructions  
   import asyncio

   \# Define Specialist Agents  
   @function\_tool  
   def get\_account\_details(user\_id: str) \-\> dict:  
       """Fetches account details for a given user ID."""  
       \# Dummy implementation  
       return {"user\_id": user\_id, "balance": 100.00, "status": "active"}

   account\_agent \= Agent(  
       name="AccountAgent",  
       instructions="You handle account-related queries using the get\_account\_details tool.",  
       tools=\[get\_account\_details\]  
   )  
   knowledge\_agent \= Agent(  
       name="KnowledgeAgent",  
       instructions="You answer product FAQs using the FileSearchTool.",  
       tools=)\] \# Replace with actual ID  
   )  
   general\_query\_agent \= Agent(  
       name="GeneralQueryAgent",  
       instructions="You answer general questions using WebSearchTool.",  
       tools=  
   )

   \# Define Triage Agent  
   triage\_agent \= Agent(  
       name="TriageAgent",  
       instructions=prompt\_with\_handoff\_instructions(  
           "You are a customer support triage agent. "  
           "Route queries to AccountAgent for account issues, "  
           "KnowledgeAgent for product questions, or "  
           "GeneralQueryAgent for other inquiries."  
       ),  
       handoffs=\[account\_agent, knowledge\_agent, general\_query\_agent\]  
   )

   \# async def run\_triage():  
   \#     result \= await Runner.run(triage\_agent, "What's my account balance for user\_id 123?")  
   \#     print(f"Query: What's my account balance for user\_id 123?\\nResponse: {result.final\_output}\\n")  
   \#     result \= await Runner.run(triage\_agent, "Tell me about product X.")  
   \#     print(f"Query: Tell me about product X.\\nResponse: {result.final\_output}\\n")  
   \# asyncio.run(run\_triage())

2. **Simple Research Bot (Conceptual based on** 41**)**:  
   Python  
   from agents import Agent, Runner, WebSearchTool, FileSearchTool  
   import asyncio

   research\_agent \= Agent(  
       name="ResearchBot",  
       instructions="Use available tools to research the user's query thoroughly and provide a summarized answer.",  
       tools=) \# Replace  
       \]  
   )

   \# async def run\_research():  
   \#     result \= await Runner.run(research\_agent, "What are the latest advancements in quantum computing and summarize findings from my uploaded 'quantum\_notes.pdf'?")  
   \#     print(f"Query: What are the latest advancements in quantum computing and summarize findings from my uploaded 'quantum\_notes.pdf'?\\nResponse: {result.final\_output}")  
   \# asyncio.run(run\_research())

The diverse range of these applications underscores the SDK's adaptability. Its lightweight nature and Python-first design encourage experimentation and integration into various problem domains that benefit from distributed intelligence and task-specific agent capabilities. The evolution from the experimental "Swarm" to the more production-oriented Agents SDK has clearly broadened the scope for practical, real-world multi-agent systems.

## **7\. Deployment and Operationalization**

Deploying applications built with the OpenAI Agents SDK involves standard software deployment practices, adapted for Python and potentially asynchronous applications. Considerations include containerization, serverless architectures, CI/CD pipelines, secret management, and performance optimization.

### **7.1 Containerizing Agents SDK Applications (Docker)**

Containerization using Docker is a common method for packaging and deploying Python applications, including those built with the Agents SDK, ensuring consistency across different environments.

* **General Practices**: Standard Docker best practices for Python applications apply. This includes using an official Python base image, copying application code and dependencies (typically via a requirements.txt file), setting the working directory, and defining the command to run the application.78  
* **Asyncio Considerations**: Since the Agents SDK heavily utilizes asyncio for its Runner.run() and Runner.run\_streamed() methods 3, the Docker CMD or ENTRYPOINT should correctly invoke the Python script that starts the asyncio event loop. For web-facing agent applications (e.g., if wrapped with FastAPI or AIOHTTP), considerations for ASGI servers like Uvicorn within Docker are relevant. Setting PYTHONUNBUFFERED=1 is often recommended for better logging from within containers.79  
* **Dependency Management**: A requirements.txt file should list openai-agents and any other necessary packages.  
* **Environment Variables**: API keys (e.g., OPENAI\_API\_KEY) and other configurations should be passed as environment variables to the Docker container at runtime, not hardcoded into the image.22

A sample Dockerfile for an Agents SDK application might look like:

Dockerfile

\# Use an official Python runtime as a parent image  
FROM python:3.10\-slim

\# Set the working directory in the container  
WORKDIR /app

\# Set environment variables for unbuffered Python output  
ENV PYTHONUNBUFFERED=1

\# Copy the dependencies file to the working directory  
COPY requirements.txt.

\# Install any needed packages specified in requirements.txt  
RUN pip install \--no-cache-dir \-r requirements.txt

\# Copy the content of the local src directory to the working directory  
COPY..

\# Command to run the application  
\# Replace your\_agent\_app.py with the entry point of your application  
\# Ensure OPENAI\_API\_KEY and other necessary environment variables are set when running the container  
CMD \["python", "your\_agent\_app.py"\]

While specific examples of Dockerizing Agents SDK applications are not extensively detailed in the provided material beyond general Python app advice, tracing tool integrations like Arize Phoenix sometimes use Docker for their own components.66 Google ADK also mentions containerization as a deployment option.81 Cloud Run, a platform that can deploy Agents SDK apps, directly supports container images.82

### **7.2 Serverless Deployment**

Serverless platforms offer scalable and cost-effective ways to deploy Agents SDK applications, particularly for event-driven or intermittently used agents.

**7.2.1 AWS Lambda**

* **Async Handlers**: AWS Lambda supports Python runtimes. For asyncio-based applications like those using Agents SDK, the Lambda handler must correctly manage the asyncio event loop. Libraries like Mangum or Lynara can adapt ASGI applications (e.g., FastAPI wrapping an agent endpoint) to Lambda's handler format.83  
* **Dependencies**: Dependencies need to be packaged in a deployment zip file or a Lambda layer.  
* **Timeouts and Resources**: Lambda functions have execution time limits and configurable memory. Complex agent interactions involving multiple LLM calls or long-running tools might hit these limits, requiring optimization or strategies like breaking down tasks.  
* **Use Cases**: Event-triggered agents (e.g., processing a new file in S3) or agents exposed via API Gateway. The aws-samples/bedrock-access-gateway project, for instance, uses Lambda to provide an OpenAI-compatible endpoint for Bedrock models, which could conceptually be adapted for Agents SDK if exposing it via an OpenAI-like API.87 The Agent Squad framework also details Lambda deployment for its Python agents.88

**7.2.2 Google Cloud Functions / Cloud Run**

* **Google Cloud Functions (GCF)**: Suitable for deploying individual Python functions, including those that might invoke an agent for a specific task.89 HTTP triggers are common. Similar considerations for asyncio and dependencies as AWS Lambda apply.  
* **Google Cloud Run**: A fully managed platform for deploying containerized applications.82 This is often a better fit for more complex Agents SDK applications or those requiring longer execution times or more resources than GCF allows. Since Cloud Run deploys containers, the Dockerization practices from Section 7.1 are directly applicable.  
* **Vertex AI Agent Engine**: While primarily for frameworks like Google ADK or LangChain, Vertex AI Agent Engine offers a managed runtime for deploying agents on Google Cloud.92 Deploying a custom Agents SDK application here would likely involve packaging it as a container and potentially adapting it to fit supported interfaces if direct SDK support is not available.

Serverless deployment simplifies infrastructure management but requires careful attention to execution limits, dependency packaging, and cold start performance.

### **7.3 CI/CD Pipelines for Agentic Applications**

Continuous Integration and Continuous Delivery (CI/CD) pipelines are crucial for maintaining the quality, reliability, and agility of agentic applications.

* **Core CI/CD Steps**:  
  1. **Code Commit**: Developers push code to a version control system (e.g., Git).  
  2. **Build**: The pipeline automatically builds the application, which might include installing dependencies and creating Docker images.  
  3. **Testing**: This is a critical and nuanced phase for agentic systems.  
     * **Unit Tests**: For individual tools and helper functions.  
     * **Integration Tests**: For interactions between agents or agents and tools.  
     * **Behavioral/Evaluation Tests**: Testing the agent's decision-making, response quality, and adherence to instructions. This often involves predefined test cases, prompt datasets, and potentially using LLM-as-judge methodologies.59 The built-in tracing and evaluation capabilities of the Agents SDK can be leveraged here.  
  4. **Deployment**: Automatically deploy to staging and production environments upon successful testing.  
* **Platform Specifics**:  
  * **Azure**: Azure DevOps can be used to set up CI/CD pipelines for Agents SDK applications, deploying to Azure Container Apps, Azure Functions, or Azure Kubernetes Service (AKS).59  
  * **GitHub Actions**: Can be used for building and testing Python applications, including those using the Agents SDK. The "Swarms" framework by Kye Gomez (distinct from OpenAI's original Swarm concept but sharing the name) shows GitHub Actions for CI.97  
  * **Google Cloud**: Cloud Build can automate builds and deployments to Cloud Run or GCF.82  
* **Challenges**: Testing the non-deterministic behavior of LLM-driven agents requires specialized evaluation strategies beyond traditional software testing. Monitoring agent performance and behavior in production is also key for iterative improvement. The VoicePipeline in the Agents SDK 98 and integrations like Docs AI search with CI/CD logs 99 hint at the complexity and specialized needs of agentic CI/CD.

### **7.4 Managing Secrets and API Keys Securely**

Secure management of API keys (e.g., OPENAI\_API\_KEY, keys for tools accessing external services like Stripe 27) and other secrets is paramount.

* **Avoid Hardcoding**: Secrets should never be hardcoded into source code.22  
* **Environment Variables**: For local development and some deployment scenarios, environment variables are a common way to provide secrets to the application.4 The Agents SDK examples frequently assume OPENAI\_API\_KEY is set this way.  
* **Secret Management Services**: For production environments, dedicated secret management services are highly recommended. These include:  
  * AWS Secrets Manager  
  * Google Secret Manager 89  
  * Azure Key Vault 59 These services provide secure storage, access control (IAM), auditing, and often versioning/rotation capabilities.  
* **Application Access**: Applications running in cloud environments should use IAM roles or service accounts with appropriately scoped permissions to access secrets from these services. The MCP extension for Agents SDK also allows secrets to be defined in a separate mcp\_agent.secrets.yaml file or via environment variables.35

### **7.5 Performance Considerations and Optimization**

Performance in multi-agent systems built with the Agents SDK is influenced by several factors:

* **LLM Inference Latency**: The choice of LLM significantly impacts performance. Larger, more capable models generally have higher latency than smaller, faster ones.9  
* **Tool Execution Latency**: Tools that call external APIs or perform complex computations can add significant delays.  
* **Number of Turns/Agents**: Each agent interaction or LLM call in the agent loop adds to the overall latency. Complex workflows with many turns or handoffs will naturally be slower.100  
* **Network Latency**: If agents or tools communicate over a network, this can introduce delays.

**Optimization Strategies**:

* **Model Selection**: Use the most capable model necessary for a task. For simpler tasks (e.g., triage, simple tool parameter extraction), smaller, faster models (like gpt-3.5-turbo or gpt-4o-mini) might suffice, reserving larger models for complex reasoning or generation.9  
* **Asynchronous Execution**: Leverage the SDK's asynchronous capabilities (Runner.run(), Runner.run\_streamed()) to handle I/O-bound operations (like tool calls to external APIs) without blocking the main execution thread, improving responsiveness.9 For tasks that can run independently, asyncio.gather can be used for parallel execution.31  
* **Prompt Engineering**: Well-crafted prompts can lead to more efficient LLM responses, potentially reducing the number of turns or tool calls needed.  
* **Tool Optimization**: Ensure custom tools are efficient and minimize external API call latency.  
* **Caching**: For deterministic tool calls or frequently accessed data, implement caching strategies to avoid redundant computations or API requests.  
* **Streaming**: For user-facing applications, use Runner.run\_streamed() to provide incremental output to the user, improving perceived performance even if the total execution time is long.3  
* **Minimize Agent Hops**: While multi-agent systems offer modularity, excessive handoffs for trivial tasks can increase overhead. Design workflows to balance specialization with efficiency.

Agentic systems often involve a trade-off between performance (latency, cost) and task capability.96 Continuous monitoring and profiling using tracing tools are essential for identifying bottlenecks and optimizing performance.

### **7.6 Table 7.1: Deployment Options Summary for OpenAI Agents SDK Applications**

| Platform | Pros | Cons/Considerations |
| :---- | :---- | :---- |
| **Docker (Self-hosted/Managed)** | High flexibility, environment consistency, portability, full control over resources. Suitable for complex, long-running agents. | Requires infrastructure management (if self-hosted), resource provisioning, scaling setup. |
| **AWS Lambda** | Cost-effective (pay-per-use), automatic scaling, event-driven capabilities, integration with AWS ecosystem. | Cold starts, execution time limits (max 15 min), package size limits, asyncio handler complexity. May require API Gateway for HTTP exposure. |
| **Google Cloud Functions** | Cost-effective (pay-per-use), automatic scaling, event-driven, integration with GCP ecosystem. | Cold starts, execution time limits (max 9 min for HTTP, up to 60 min for event-driven 2nd gen), asyncio handler complexity. |
| **Google Cloud Run** | Fully managed, scales automatically (to zero), supports containers, longer execution times than GCF, direct HTTP(S) endpoints. | Can have cold starts (though configurable), potentially higher cost than GCF for very infrequent invocations if min instances \> 0\. |
| **Azure (Container Apps, Functions, AKS)** | Variety of options from serverless (Functions) to container orchestration (AKS, Container Apps), strong enterprise integration. | Similar considerations to AWS/GCP counterparts (cold starts for Functions, complexity for AKS). |

This table provides a high-level comparison to aid in selecting an appropriate deployment strategy based on specific project needs, operational capabilities, and existing cloud infrastructure.

## **8\. Comparative Analysis: OpenAI Agents SDK in the Ecosystem**

The landscape of AI agent orchestration frameworks has become increasingly populated, with several notable alternatives to the OpenAI Agents SDK. Understanding their differences in terms of design philosophy, features, and ideal use cases is crucial for developers selecting the right tool. This section compares the OpenAI Agents SDK with LangChain/LangGraph, CrewAI, AutoGen, and Google's Agent Development Kit (ADK).

### **8.1 OpenAI Agents SDK vs. LangChain/LangGraph**

* **OpenAI Agents SDK**:  
  * **Focus**: Simplicity, Python-first design with minimal new abstractions, tight integration with OpenAI models and the Responses API.3  
  * **Strengths**: Easy to get started, lightweight, strong built-in tracing and guardrails, automatic schema generation for tools.3  
  * **Orchestration**: Relies on an agent loop, handoffs for delegation, and Pythonic control flow.3  
  * **State Management**: Primarily through passing conversation history; persistent state requires external tools.4  
* **LangChain/LangGraph**:  
  * **Focus**: LangChain is a comprehensive framework for building LLM applications, offering a wide array of components (models, prompts, memory, indexes, chains, agents). LangGraph is a library built on LangChain for creating stateful, multi-actor applications, particularly agents, by representing them as graphs.77  
  * **Strengths**: Extensive ecosystem and integrations, powerful for complex and cyclical stateful workflows (LangGraph provides full authorship over cognitive architecture 95), explicit state management, and robust debugging with LangSmith.48 LangGraph offers pre-built templates for common agent architectures, including one inspired by Swarm.95  
  * **Orchestration**: LangGraph uses a graph-based approach where nodes represent functions (often LLM calls or tools) and edges represent conditional transitions, allowing for explicit control over the flow and state.95  
  * **State Management**: LangGraph excels at managing state explicitly within the graph structure, supporting both short-term working memory and persistent long-term memory.57  
  * **Learning Curve**: Generally considered steeper than the OpenAI Agents SDK due to its more extensive set of abstractions and graph-based concepts.48

Comparison Insights:  
The OpenAI Agents SDK prioritizes simplicity and a native Python feel, aiming for a lower barrier to entry, especially for developers already familiar with OpenAI APIs. It provides core primitives for multi-agent orchestration with less prescriptive structure. LangGraph, in contrast, offers a more structured and powerful (but also more complex) approach for building agents with explicit state management and control over complex, potentially cyclical, workflows. While OpenAI's official guide on building agents has been critiqued by LangChain developers for some of its takes 96, the existence of a "Swarm" pre-built in LangGraph suggests some conceptual alignment but differing implementation philosophies regarding control and abstraction.

### **8.2 OpenAI Agents SDK vs. CrewAI**

* **OpenAI Agents SDK**: General-purpose framework for defining agents with instructions, tools, and handoffs.  
* **CrewAI**:  
  * **Focus**: Orchestrating role-playing autonomous AI agents that collaborate as a "crew" to achieve complex objectives.77 Emphasizes human-agent synergy and an intuitive design, potentially for less technical users in defining agent roles and tasks.48  
  * **Strengths**: Intuitive role-based design (agents have specific roles, backstories, goals), straightforward setup for collaborative tasks, good for automating team-like workflows.48  
  * **Orchestration**: Agents collaborate within a "crew," with tasks often processed sequentially or hierarchically based on defined roles and goals.108  
  * **State Management**: Basic state persistence through task outputs.48  
  * **Learning Curve**: Considered relatively easy to get started with, second only to the OpenAI Agents SDK in one comparison.48

Comparison Insights:  
CrewAI offers a higher-level abstraction focused on a specific paradigm: collaborative "crews" of agents with clearly defined roles. This makes it very intuitive for tasks that map well to human team structures. The OpenAI Agents SDK is more foundational, providing the building blocks (agents, tools, handoffs) for various orchestration patterns without enforcing a specific collaborative model like "crews." CrewAI might be preferred for rapid development of role-based multi-agent systems, while the Agents SDK offers more flexibility for custom architectures.

### **8.3 OpenAI Agents SDK vs. AutoGen**

* **OpenAI Agents SDK**: Focuses on an agent loop with tools, handoffs, and guardrails for task execution and delegation.  
* **AutoGen (Microsoft Research)**:  
  * **Focus**: A framework for simplifying the orchestration, optimization, and automation of complex LLM workflows. It excels in enabling flexible conversation patterns and diverse agent topologies.8  
  * **Strengths**: Strong support for multi-agent conversations, allowing complex interactions and collaborations between multiple (AI and human) agents. Capable of implementing various design patterns, including handoffs.110  
  * **Orchestration**: Based on "conversable agents" that communicate with each other to solve tasks. Supports various predefined and customizable conversation patterns.  
  * **State Management**: Agents can maintain memory, suitable for conversation-driven workflows.48  
  * **Learning Curve**: Described as moderately tricky, with some documentation and versioning clarity issues noted in one review.48

Comparison Insights:  
AutoGen appears to offer more specialized and fine-grained control over the dynamics of multi-agent conversations and interactions. Its strength lies in defining how agents communicate and collaborate through various chat patterns. The OpenAI Agents SDK, while supporting multi-agent systems through handoffs and agents-as-tools, provides a more general agent execution loop. AutoGen might be preferred for applications where the conversational aspect and complex inter-agent dialogues are central, whereas the Agents SDK offers a more direct approach to task delegation and tool execution within a simpler set of primitives.

### **8.4 OpenAI Agents SDK vs. Google Agent Development Kit (ADK) / Vertex AI Agent Builder**

* **OpenAI Agents SDK**: Lightweight, Python-first, strong integration with OpenAI's ecosystem, particularly the Responses API.  
* **Google Agent Development Kit (ADK)**:  
  * **Focus**: A flexible, modular framework optimized for the Google Cloud ecosystem (Gemini models, Vertex AI) but designed to be model-agnostic and deployment-agnostic.76 Aims to make agent development feel more like traditional software development.  
  * **Strengths**: Rich tool ecosystem (including pre-built Google product integrations, MCP tools), built-in evaluation, robust guardrail features, easy deployment to Vertex AI Agent Engine, support for workflow agents (Sequential, Parallel, Loop) and LLM-driven dynamic routing.81 Good for enterprise scenarios and multimodal agents.115  
  * **Vertex AI Agent Builder**: A fully managed Google Cloud service for deploying, managing, and scaling AI agents built with ADK, LangChain, and other frameworks.92

Comparison Insights:  
Google ADK is positioned as a comprehensive, enterprise-grade framework with deep integration into Google Cloud, offering extensive tooling and managed deployment options via Vertex AI Agent Engine. It appears more feature-rich in terms of built-in enterprise connectors and evaluation capabilities. The OpenAI Agents SDK is more lightweight and focused on providing core primitives for agent orchestration with a strong emphasis on simplicity and the OpenAI ecosystem. The choice often depends on the preferred cloud ecosystem, the need for specific Google Cloud integrations, or the preference for OpenAI's more streamlined, Python-native approach. Some community discussions suggest ADK might have an edge for tasks requiring deep Google ecosystem integration, while the OpenAI SDK offers more control over what is shared (streamed) and potentially lower latency for OpenAI models.113

### **8.5 Strengths, Weaknesses, and Ideal Use Cases for Each**

| Framework | Strengths | Weaknesses/Considerations | Ideal Use Cases |
| :---- | :---- | :---- | :---- |
| **OpenAI Agents SDK** | Lightweight, Python-first, minimal abstractions, easy to learn, strong OpenAI integration, built-in tracing & guardrails, Responses API. 3 | Newer, ecosystem still growing, persistent memory is developer-managed. Limited built-in parallel execution compared to graph-based systems. 4 | Rapid prototyping, developers preferring simplicity and OpenAI ecosystem, applications needing straightforward multi-agent delegation and tool use, production-ready focus. |
| **LangGraph** | Powerful for complex stateful/cyclical workflows, explicit state control, strong LangChain ecosystem, robust debugging (LangSmith). 48 | Steeper learning curve, more abstractions to learn (graph concepts). 48 | Complex agentic systems requiring fine-grained control over state and execution flow, applications already using LangChain. |
| **CrewAI** | Intuitive role-based agent definition, easy setup for collaborative tasks, good for human-agent synergy, user-friendly. 48 | May be less flexible for non-role-based architectures, production readiness for diverse scenarios still maturing. 48 | Automating team-like tasks, scenarios where clear agent roles map to human workflows, projects needing quick setup of collaborative agents. |
| **AutoGen** | Strong for diverse multi-agent conversation patterns, flexible agent topologies, backed by Microsoft Research. 48 | Can be complex to set up, documentation clarity has been a concern for some. 48 | Applications centered around complex multi-agent dialogues, research in conversational AI, scenarios requiring sophisticated inter-agent communication protocols. |
| **Google ADK** | Deep Google Cloud/Vertex AI integration, rich tool ecosystem (incl. enterprise connectors), built-in evaluation, robust guardrails. 81 | Can be heavily dependent on Google ecosystem for optimal use, potentially less intuitive for simple scenarios if not using ADK Web. 115 | Enterprise applications within the Google Cloud ecosystem, tasks requiring strong integration with Google services, multimodal applications, production-grade deployments on Vertex AI. |

No single framework is universally superior. The optimal choice depends on the specific project requirements, the development team's expertise, the desired level of control versus abstraction, and the target deployment ecosystem. The OpenAI Agents SDK carves out a niche by offering a balance of simplicity, power, and tight integration with OpenAI's evolving platform capabilities.

## **9\. Roadmap and Future Developments (as of May 26, 2025\)**

OpenAI is actively investing in its agent-building tools, with a clear roadmap focused on enhancing capabilities, simplifying development, and providing a seamless platform experience for creating autonomous systems. Key future developments revolve around the Agents SDK and the Responses API.39

**Agents SDK Development:**

* **Open-Source Commitment**: OpenAI plans to continue developing the Agents SDK as an open-source framework, encouraging community contributions and expansion upon their approach.39  
* **Language Support**: While initially launched with Python support, Node.js support for the Agents SDK is planned, broadening its accessibility to JavaScript developers.39 Community efforts have already produced TypeScript variants like openai-agents-js.119  
* **Feature Enhancements**: Ongoing improvements are expected for core features like Agents (configurability, tools), Handoffs (intelligent control transfer), Guardrails (safety checks), and Tracing & Observability (debugging and performance optimization).39

**Responses API Evolution:**

* **Strategic Direction**: The Responses API is positioned as the future foundation for building agents on OpenAI's platform, designed to be more flexible and powerful than previous APIs for agentic applications.39  
* **Feature Parity with Assistants API**: A key roadmap item is to achieve full feature parity between the Responses API and the existing Assistants API. This includes adding support for:  
  * Assistant-like and Thread-like objects.39  
  * The **Code Interpreter** tool.39  
* **New Built-in Tools**: The Responses API will continue to integrate and support new built-in tools, expanding on the initial offerings of web search, file search, and computer use.39  
* **Usability Improvements**: Enhancements such as a unified item-based design, simpler polymorphism, intuitive streaming events, and SDK helpers (e.g., response.output\_text) are part of its design.39

**Assistants API Deprecation:**

* **Timeline**: Once the Responses API achieves full feature parity with the Assistants API, OpenAI plans to formally announce the deprecation of the Assistants API. The target sunset date for the Assistants API is **mid-2026**.39  
* **Migration**: A clear migration guide will be provided to help developers transition their applications and preserve data from the Assistants API to the Responses API.39 Until the formal deprecation, new models will continue to be delivered to the Assistants API.

Broader Vision for Agentic AI:  
OpenAI views agents as becoming integral to the workforce, enhancing productivity across various industries.39 As LLM capabilities become more agentic, OpenAI intends to continue investing in:

* Deeper integrations across their APIs.  
* New tools to help deploy, evaluate, and optimize agents in production. The overarching goal is to provide developers with a comprehensive and seamless platform for building impactful autonomous systems. The development of tools like the Agents SDK and the Responses API are central to this vision, aiming to simplify the creation of complex, multi-step, tool-using agentic applications.

## **10\. Limitations and Community Feedback (as of May 26, 2025\)**

While the OpenAI Agents SDK (and its conceptual predecessor, Swarm) offers a promising direction for multi-agent orchestration, it is important to acknowledge its limitations and consider feedback from the developer community. As of May 26, 2025, several aspects warrant attention.

**Known Limitations:**

* **Experimental Aspects and Production Readiness**:  
  * The original "Swarm" project was explicitly labeled as experimental and not recommended for production use due to potential for significant changes and lack of long-term support guarantees.7  
  * The OpenAI Agents SDK is presented as a "production-ready upgrade".1 However, as a relatively new SDK (formally announced around March 2025 20), its capabilities in large-scale, diverse production deployments are still being validated by the broader community compared to more established frameworks.8 Some comparisons still rate its production readiness as "high" but acknowledge its newness.48  
  * The ComputerTool is still in research preview, indicating it is not yet fully production-ready.38  
* **Persistent Memory**:  
  * The Agents SDK itself does not provide built-in mechanisms for persistent memory across agent runs or sessions.4 State management for long-term memory is largely the developer's responsibility, typically handled by creating tools that interact with external storage solutions like vector databases or dedicated memory services.4 This contrasts with some frameworks that might offer more integrated memory solutions. The FileSearchTool offers a form of RAG with OpenAI-managed storage, but broader custom persistence requires external implementation.37  
* **Complexity in Certain Scenarios**:  
  * While designed for simplicity, orchestrating very complex, deeply nested, or highly dynamic multi-agent interactions can still pose challenges. Community discussions highlight difficulties in ensuring autonomous handoffs beyond the first level in certain nested scenarios, sometimes requiring explicit prompting or falling back to "agents as tools" patterns.46  
  * Performance in complex multi-agent systems, especially those involving multiple LLM calls and tool invocations per user query, can be a concern, leading to noticeable latency.100 This is a general challenge in agentic systems, not unique to the SDK, but an important consideration.  
* **Feature Parity and API Evolution**:  
  * The Responses API, central to the SDK's interaction with newer OpenAI features, is still working towards full feature parity with the older Assistants API (e.g., Code Interpreter support is planned but was not fully available initially).39 This means developers might need to wait for certain functionalities or use workarounds.  
  * As a rapidly evolving field, APIs and SDK features may change, potentially requiring adaptation by early adopters.8

**Community Feedback and Discussion Points (Observed up to May 2025):**

* **Model Compatibility and Tooling**:  
  * There have been questions and discussions around compatibility with specific models (e.g., OpenAI Realtime-preview models, which are not directly compatible due to different endpoint requirements 121) and Azure OpenAI model endpoints.121 While workarounds exist for Azure (e.g., explicitly passing an AsyncAzureOpenAI client 122), full seamless support for all variations and hosted tools across different cloud provider versions of OpenAI models (like web search on Azure) has been a point of discussion.122  
  * Support for native tool formats from other model providers when using integrations like LiteLLM (e.g., Google Search tool for Gemini) has been raised as a feature request/clarification point.123  
* **Developer Experience**:  
  * The Python-first approach and minimal abstractions are generally well-received for ease of learning.48  
  * Requests for a TypeScript/JavaScript version of the Agents SDK have been prominent, with community members even starting their own implementations.119 OpenAI has acknowledged plans for Node.js support.39  
  * Minor bugs or unexpected behaviors with specific features like the @function\_tool decorator when used outside agent logic for debugging have been reported.43  
* **Ecosystem and Comparisons**:  
  * Active comparisons are being made with other frameworks like LangGraph, CrewAI, AutoGen, and Google ADK, with discussions focusing on trade-offs in complexity, features, state management, and ease of use.48 This reflects a community actively evaluating the SDK's place in the broader ecosystem.  
* **Documentation and Examples**:  
  * The official documentation and examples are generally seen as helpful for getting started.48 However, as with any new SDK, there are ongoing needs for more advanced examples and clarifications on specific edge cases or integrations.124

Community channels like the OpenAI Developer Forum serve as important platforms for these discussions, bug reports, and feature requests.43 The active engagement suggests a strong interest in the SDK and its potential, alongside a collaborative effort to refine and expand its capabilities.

## **11\. Conclusion**

The OpenAI Agents SDK, emerging from the experimental "Swarm" project, represents a significant advancement in the toolkit available for developing lightweight and effective multi-agent orchestration systems. Between its formalization in early 2025 and May 2025, the SDK has established itself as a production-ready framework characterized by its Python-first design, minimal abstractions, and a core set of powerful primitives: Agents, Tools, Handoffs, and Guardrails. Its tight integration with OpenAI's evolving model ecosystem, particularly through the Responses API, and its extensibility to other LLM providers via mechanisms like LiteLLM, position it as a versatile solution for a wide range of developers.

The SDK's architecture, centered around a managed agent loop, simplifies the complexities of iterative LLM interactions, tool execution, and inter-agent delegation. Features such as automatic schema generation for function tools, built-in configurable guardrails for safety, and comprehensive tracing capabilities significantly enhance developer ergonomics and facilitate the creation of more reliable and debuggable agentic applications. The provision for both LLM-driven and code-driven orchestration allows developers to choose the appropriate level of autonomy and predictability for their specific use cases, which span from sophisticated customer service systems and research assistants to automated content generation and task management.

However, the SDK is not without its considerations. Persistent memory management remains largely a developer responsibility, requiring integration with external storage solutions. While the framework is designed for production, its relative newness means that community best practices for large-scale, complex deployments are still maturing. Performance in highly intricate multi-agent scenarios and seamless compatibility across all features with third-party model endpoints (especially those hosted on different cloud platforms like Azure) continue to be areas of active development and community feedback.

The roadmap indicates a strong commitment from OpenAI to further enhance the Agents SDK and the underlying Responses API, with plans for expanded language support (Node.js) and richer functionalities, including the integration of tools like Code Interpreter into the Responses API. The planned deprecation of the Assistants API in mid-2026 in favor of the Responses API underscores the strategic importance of this new interaction paradigm for

#### **Works cited**

1. OpenAI Open-Sources Agents SDK: Building Agentic AI Applications, accessed May 21, 2025, [https://learnprompting.org/blog/openai-agents-sdk](https://learnprompting.org/blog/openai-agents-sdk)  
2. OpenAI Agents SDK \- Humanloop, accessed May 21, 2025, [https://humanloop.com/blog/openai-agents-sdk](https://humanloop.com/blog/openai-agents-sdk)  
3. openai/openai-agents-python: A lightweight, powerful framework for multi-agent workflows, accessed May 21, 2025, [https://github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python)  
4. OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/](https://openai.github.io/openai-agents-python/)  
5. Open AI Agents SDK : Unlocking the Future of OpenAI's Game-Changing Agents SDK Update Part-1 \- DEV Community, accessed May 21, 2025, [https://dev.to/sreeni5018/open-ai-agents-sdk-unlocking-the-future-of-openais-game-changing-agents-sdk-update-part-1-eai](https://dev.to/sreeni5018/open-ai-agents-sdk-unlocking-the-future-of-openais-game-changing-agents-sdk-update-part-1-eai)  
6. Introducing OpenAI's Swarm: A New Open-Source Multi-Agent ..., accessed May 21, 2025, [https://www.kommunicate.io/blog/openai-swarm/](https://www.kommunicate.io/blog/openai-swarm/)  
7. How OpenAI Swarm Enhances Multi-Agent Collaboration? \- Analytics Vidhya, accessed May 21, 2025, [https://www.analyticsvidhya.com/blog/2024/10/openai-swarm/](https://www.analyticsvidhya.com/blog/2024/10/openai-swarm/)  
8. Swarm: The Agentic Framework from OpenAI \- Fluid AI, accessed May 21, 2025, [https://www.fluid.ai/blog/swarm-the-agentic-framework-from-openai](https://www.fluid.ai/blog/swarm-the-agentic-framework-from-openai)  
9. How to Use the OpenAI Agents SDK ? \- Apidog, accessed May 21, 2025, [https://apidog.com/blog/how-to-use-openai-agents-sdk/](https://apidog.com/blog/how-to-use-openai-agents-sdk/)  
10. Building Voice AI Agents with the OpenAI Agents SDK \- DEV Community, accessed May 21, 2025, [https://dev.to/cloudx/building-voice-ai-agents-with-the-openai-agents-sdk-2aog](https://dev.to/cloudx/building-voice-ai-agents-with-the-openai-agents-sdk-2aog)  
11. agents-sdk-models \- PyPI, accessed May 21, 2025, [https://pypi.org/project/agents-sdk-models/0.0.13/](https://pypi.org/project/agents-sdk-models/0.0.13/)  
12. cookbook/gen-ai/openai/agents-sdk-intro.ipynb at main · aurelio ..., accessed May 21, 2025, [https://github.com/aurelio-labs/cookbook/blob/main/gen-ai/openai/agents-sdk-intro.ipynb](https://github.com/aurelio-labs/cookbook/blob/main/gen-ai/openai/agents-sdk-intro.ipynb)  
13. Quickstart \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/quickstart/](https://openai.github.io/openai-agents-python/quickstart/)  
14. Building Agentic AI Applications using OpenAI Agents SDK \- Association of Data Scientists, accessed May 21, 2025, [https://adasci.org/building-agentic-ai-applications-using-openai-agents-sdk/](https://adasci.org/building-agentic-ai-applications-using-openai-agents-sdk/)  
15. Projects with the OpenAI Agents SDK \- RIIS LLC, accessed May 21, 2025, [https://www.riis.com/blog/projects-with-the-openai-agents-sdk](https://www.riis.com/blog/projects-with-the-openai-agents-sdk)  
16. A Deep Dive Into The OpenAI Agents SDK \- Sid Bharath, accessed May 21, 2025, [https://www.siddharthbharath.com/openai-agents-sdk/](https://www.siddharthbharath.com/openai-agents-sdk/)  
17. Using any model via LiteLLM \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/models/litellm/](https://openai.github.io/openai-agents-python/models/litellm/)  
18. Models \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/models/](https://openai.github.io/openai-agents-python/models/)  
19. OpenAI Agents SDK Tutorial: Building AI Systems That Take Action | DataCamp, accessed May 21, 2025, [https://www.datacamp.com/tutorial/openai-agents-sdk-tutorial](https://www.datacamp.com/tutorial/openai-agents-sdk-tutorial)  
20. OpenAI Agents SDK: A Step-by-Step Guide to Building Real-World MCP Agents with Composio \- DEV Community, accessed May 21, 2025, [https://dev.to/composiodev/openai-agents-sdk-a-step-by-step-guide-to-building-real-world-mcp-agents-with-composio-4f92](https://dev.to/composiodev/openai-agents-sdk-a-step-by-step-guide-to-building-real-world-mcp-agents-with-composio-4f92)  
21. Developer quickstart \- OpenAI API, accessed May 21, 2025, [https://platform.openai.com/docs/quickstart](https://platform.openai.com/docs/quickstart)  
22. How to Use and Get OpenAI API Key: Complete Guide \- Elfsight, accessed May 21, 2025, [https://elfsight.com/blog/how-to-use-open-ai-api-key/](https://elfsight.com/blog/how-to-use-open-ai-api-key/)  
23. API Reference \- OpenAI Platform, accessed May 21, 2025, [https://platform.openai.com/docs/api-reference/introduction](https://platform.openai.com/docs/api-reference/introduction)  
24. OpenAI Agents SDK \- AgentOps, accessed May 21, 2025, [https://docs.agentops.ai/v1/integrations/agentssdk](https://docs.agentops.ai/v1/integrations/agentssdk)  
25. openai-agents-python/examples/basic/hello\_world\_jupyter.py at main \- GitHub, accessed May 21, 2025, [https://github.com/openai/openai-agents-python/blob/main/examples/basic/hello\_world\_jupyter.py](https://github.com/openai/openai-agents-python/blob/main/examples/basic/hello_world_jupyter.py)  
26. openai-agents-python/examples/agent\_patterns/routing.py at main \- GitHub, accessed May 21, 2025, [https://github.com/openai/openai-agents-python/blob/main/examples/agent\_patterns/routing.py](https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/routing.py)  
27. Automating Dispute Management with Agents SDK and Stripe API | OpenAI Cookbook, accessed May 21, 2025, [https://cookbook.openai.com/examples/agents\_sdk/dispute\_agent](https://cookbook.openai.com/examples/agents_sdk/dispute_agent)  
28. OpenAI Agents SDK — Getting Started \- GetStream.io, accessed May 21, 2025, [https://getstream.io/blog/openai-agents-sdk/](https://getstream.io/blog/openai-agents-sdk/)  
29. Running agents \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/running\_agents/](https://openai.github.io/openai-agents-python/running_agents/)  
30. OpenAI \- A practical guide to building agents, accessed May 21, 2025, [https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)  
31. Orchestrating multiple agents \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/multi\_agent/](https://openai.github.io/openai-agents-python/multi_agent/)  
32. OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/ref/agent/](https://openai.github.io/openai-agents-python/ref/agent/)  
33. openai-agents-python/src/agents/agent.py at main \- GitHub, accessed May 21, 2025, [https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/agent.py)  
34. Model settings \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/ref/model\_settings/](https://openai.github.io/openai-agents-python/ref/model_settings/)  
35. lastmile-ai/openai-agents-mcp: An MCP extension package for OpenAI Agents SDK \- GitHub, accessed May 21, 2025, [https://github.com/lastmile-ai/openai-agents-mcp](https://github.com/lastmile-ai/openai-agents-mcp)  
36. Lifecycle \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/ref/lifecycle/](https://openai.github.io/openai-agents-python/ref/lifecycle/)  
37. Tools \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/tools/](https://openai.github.io/openai-agents-python/tools/)  
38. Deep Dive into OpenAI's Agent Ecosystem \- Gradient Flow, accessed May 21, 2025, [https://gradientflow.com/deep-dive-into-openais-agent-ecosystem/](https://gradientflow.com/deep-dive-into-openais-agent-ecosystem/)  
39. New tools for building agents | OpenAI, accessed May 21, 2025, [https://openai.com/index/new-tools-for-building-agents/](https://openai.com/index/new-tools-for-building-agents/)  
40. OpenAI launches Responses API and SDK for AI agent workflows \- ContentGrip, accessed May 21, 2025, [https://www.contentgrip.com/openai-new-tools-for-building-ai-agents/](https://www.contentgrip.com/openai-new-tools-for-building-ai-agents/)  
41. Building a Voice Assistant with the Agents SDK | OpenAI Cookbook, accessed May 21, 2025, [https://cookbook.openai.com/examples/agents\_sdk/app\_assistant\_voice\_agents](https://cookbook.openai.com/examples/agents_sdk/app_assistant_voice_agents)  
42. Tools \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/ref/tool/](https://openai.github.io/openai-agents-python/ref/tool/)  
43. AgentsSDK \- @function\_tool decorator throws Type Error when used as a regular function for debugging \- Bugs \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/agentssdk-function-tool-decorator-throws-type-error-when-used-as-a-regular-function-for-debugging/1158507](https://community.openai.com/t/agentssdk-function-tool-decorator-throws-type-error-when-used-as-a-regular-function-for-debugging/1158507)  
44. Top Openai Swarm features \- APPWRK, accessed May 21, 2025, [https://appwrk.com/open-ai-swarm-features](https://appwrk.com/open-ai-swarm-features)  
45. Handoffs \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/handoffs/](https://openai.github.io/openai-agents-python/handoffs/)  
46. Handoff to multiple agents and transfer back to triage agent · Issue \#256 \- GitHub, accessed May 21, 2025, [https://github.com/openai/openai-agents-python/issues/256](https://github.com/openai/openai-agents-python/issues/256)  
47. OpenAI's Agents SDK and Anthropic's Model Context Protocol (MCP) \- PromptHub, accessed May 21, 2025, [https://www.prompthub.us/blog/openais-agents-sdk-and-anthropics-model-context-protocol-mcp](https://www.prompthub.us/blog/openais-agents-sdk-and-anthropics-model-context-protocol-mcp)  
48. OpenAI Agents SDK vs LangGraph vs Autogen vs CrewAI \- Composio, accessed May 21, 2025, [https://composio.dev/blog/openai-agents-sdk-vs-langgraph-vs-autogen-vs-crewai/](https://composio.dev/blog/openai-agents-sdk-vs-langgraph-vs-autogen-vs-crewai/)  
49. Guardrails \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/guardrails/](https://openai.github.io/openai-agents-python/guardrails/)  
50. Guardrails \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/ref/guardrail/](https://openai.github.io/openai-agents-python/ref/guardrail/)  
51. Guardrails · Issue \#1197 · pydantic/pydantic-ai \- GitHub, accessed May 21, 2025, [https://github.com/pydantic/pydantic-ai/issues/1197](https://github.com/pydantic/pydantic-ai/issues/1197)  
52. Runner \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/ref/run/](https://openai.github.io/openai-agents-python/ref/run/)  
53. Context management \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/context/](https://openai.github.io/openai-agents-python/context/)  
54. Build AI Agents with Memory: LangChain \+ FalkorDB, accessed May 21, 2025, [https://www.falkordb.com/blog/building-ai-agents-with-memory-langchain/](https://www.falkordb.com/blog/building-ai-agents-with-memory-langchain/)  
55. Building a Memory Agent with the OpenAI Agents SDK and Zep, accessed May 21, 2025, [https://www.getzep.com/blog/building-a-memory-agent-with-the-openai-agents-sdk-and-zep/](https://www.getzep.com/blog/building-a-memory-agent-with-the-openai-agents-sdk-and-zep/)  
56. Mem0 with OpenAI Agents SDK for Voice, accessed May 21, 2025, [https://docs.mem0.ai/examples/mem0-openai-voice-demo](https://docs.mem0.ai/examples/mem0-openai-voice-demo)  
57. LangMem SDK for agent long-term memory \- LangChain Blog, accessed May 21, 2025, [https://blog.langchain.dev/langmem-sdk-launch/](https://blog.langchain.dev/langmem-sdk-launch/)  
58. Introducing Strands Agents, an Open Source AI Agents SDK \- AWS, accessed May 21, 2025, [https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)  
59. Enterprise Application Development with Azure Responses API and Agents SDK, accessed May 21, 2025, [https://techcommunity.microsoft.com/blog/azure-ai-services-blog/enterprise-application-development-with-azure-responses-api-and-agents-sdk/4393969](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/enterprise-application-development-with-azure-responses-api-and-agents-sdk/4393969)  
60. New tools for building agents | OpenAI, accessed May 21, 2025, [https://openai.com/blog/new-tools-for-building-agents](https://openai.com/blog/new-tools-for-building-agents)  
61. How To Run OpenAI Agents SDK Locally With 100+ LLMs, and ..., accessed May 21, 2025, [https://getstream.io/blog/local-openai-agents/](https://getstream.io/blog/local-openai-agents/)  
62. Tracing OpenAI \- MLflow, accessed May 21, 2025, [https://www.mlflow.org/docs/latest/tracing/integrations/openai](https://www.mlflow.org/docs/latest/tracing/integrations/openai)  
63. Trace the OpenAI Agents SDK with Langfuse, accessed May 21, 2025, [https://langfuse.com/docs/integrations/openaiagentssdk/openai-agents](https://langfuse.com/docs/integrations/openaiagentssdk/openai-agents)  
64. Example \- Tracing and Evaluation for the OpenAI-Agents SDK \- Langfuse, accessed May 21, 2025, [https://langfuse.com/docs/integrations/openaiagentssdk/example-evaluating-openai-agents](https://langfuse.com/docs/integrations/openaiagentssdk/example-evaluating-openai-agents)  
65. OpenAI Agents SDK | Arize Docs, accessed May 21, 2025, [https://docs.arize.com/arize/observe/tracing-integrations-auto/openai-agents-sdk](https://docs.arize.com/arize/observe/tracing-integrations-auto/openai-agents-sdk)  
66. OpenAI Agents SDK | Phoenix \- Arize AI, accessed May 21, 2025, [https://docs.arize.com/phoenix/tracing/integrations-tracing/openai-agents-sdk](https://docs.arize.com/phoenix/tracing/integrations-tracing/openai-agents-sdk)  
67. OpenAI Agents SDK Tracing | Phoenix \- Arize AI, accessed May 21, 2025, [https://docs.arize.com/phoenix/integrations/llm-providers/openai/openai-agents-sdk-tracing](https://docs.arize.com/phoenix/integrations/llm-providers/openai/openai-agents-sdk-tracing)  
68. Examples \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/examples/](https://openai.github.io/openai-agents-python/examples/)  
69. accessed December 31, 1969, [https://openai.github.io/openai-agents-python/api\_reference/tools/](https://openai.github.io/openai-agents-python/api_reference/tools/)  
70. OpenAI Swarm: Everything You Need To Know \- PlayHT, accessed May 21, 2025, [https://play.ht/blog/openai-swarm/](https://play.ht/blog/openai-swarm/)  
71. openai-agents-sdk · GitHub Topics, accessed May 21, 2025, [https://github.com/topics/openai-agents-sdk](https://github.com/topics/openai-agents-sdk)  
72. All You Need to Know About OpenAI Swarm: The Future of Self-Organized Autonomous AI Agents, accessed May 21, 2025, [https://felloai.com/2024/10/all-you-need-to-know-about-openai-swarm-the-future-of-self-organized-autonomous-ai-agents/](https://felloai.com/2024/10/all-you-need-to-know-about-openai-swarm-the-future-of-self-organized-autonomous-ai-agents/)  
73. OpenAI Swarm for Multi-Agent Orchestration : r/ArtificialInteligence \- Reddit, accessed May 21, 2025, [https://www.reddit.com/r/ArtificialInteligence/comments/1g1xp9n/openai\_swarm\_for\_multiagent\_orchestration/](https://www.reddit.com/r/ArtificialInteligence/comments/1g1xp9n/openai_swarm_for_multiagent_orchestration/)  
74. OpenAI Codex: Transforming Software Development with AI Agents \- DevOps.com, accessed May 21, 2025, [https://devops.com/openai-codex-transforming-software-development-with-ai-agents/](https://devops.com/openai-codex-transforming-software-development-with-ai-agents/)  
75. Using AI agents to setup your AWS infra. Usefull for developers with not much Devops expertise. \- Reddit, accessed May 21, 2025, [https://www.reddit.com/r/aws/comments/1i55d07/using\_ai\_agents\_to\_setup\_your\_aws\_infra\_usefull/](https://www.reddit.com/r/aws/comments/1i55d07/using_ai_agents_to_setup_your_aws_infra_usefull/)  
76. rawheel/Google-Agent-Development-Kit \- GitHub, accessed May 21, 2025, [https://github.com/rawheel/Google-Agent-Development-Kit](https://github.com/rawheel/Google-Agent-Development-Kit)  
77. Agent SDK vs CrewAI vs LangChain: Which One to Use When? \- Analytics Vidhya, accessed May 21, 2025, [https://www.analyticsvidhya.com/blog/2025/03/agent-sdk-vs-crewai-vs-langchain/](https://www.analyticsvidhya.com/blog/2025/03/agent-sdk-vs-crewai-vs-langchain/)  
78. Running Asyncio Web Apps with aiohttp in Docker | ProxiesAPI, accessed May 21, 2025, [https://proxiesapi.com/articles/running-asyncio-web-apps-with-aiohttp-in-docker](https://proxiesapi.com/articles/running-asyncio-web-apps-with-aiohttp-in-docker)  
79. docker compose asyncio.sleep block statements before \- Stack Overflow, accessed May 21, 2025, [https://stackoverflow.com/questions/77934209/docker-compose-asyncio-sleep-block-statements-before](https://stackoverflow.com/questions/77934209/docker-compose-asyncio-sleep-block-statements-before)  
80. asyncio client server does not work inside docker \- Stack Overflow, accessed May 21, 2025, [https://stackoverflow.com/questions/38418543/asyncio-client-server-does-not-work-inside-docker](https://stackoverflow.com/questions/38418543/asyncio-client-server-does-not-work-inside-docker)  
81. Agent Development Kit \- Google, accessed May 21, 2025, [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)  
82. Build functions into containers | Cloud Run Documentation, accessed May 21, 2025, [https://cloud.google.com/run/docs/building/functions](https://cloud.google.com/run/docs/building/functions)  
83. How to Deploy FastAPI on AWS Lambda \- YouTube, accessed May 21, 2025, [https://www.youtube.com/watch?v=b0XCH04K8eQ](https://www.youtube.com/watch?v=b0XCH04K8eQ)  
84. Simple Serverless FastAPI with AWS Lambda \- deadbearcode, accessed May 21, 2025, [https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/](https://www.deadbear.io/simple-serverless-fastapi-with-aws-lambda/)  
85. Accelize/aio\_lambda\_api: AsyncIO AWS Lambda API \- GitHub, accessed May 21, 2025, [https://github.com/Accelize/aio\_lambda\_api](https://github.com/Accelize/aio_lambda_api)  
86. mirumee/lynara: Python ASGI applications in an AWS Lambda runtime \- GitHub, accessed May 21, 2025, [https://github.com/mirumee/lynara](https://github.com/mirumee/lynara)  
87. Use Amazon Bedrock Models with OpenAI SDKs with a Serverless Proxy Endpoint \- Without Fixed Cost\! \- DEV Community, accessed May 21, 2025, [https://dev.to/aws-builders/use-amazon-bedrock-models-via-an-openai-api-compatible-serverless-endpoint-now-without-fixed-cost-5hf5](https://dev.to/aws-builders/use-amazon-bedrock-models-via-an-openai-api-compatible-serverless-endpoint-now-without-fixed-cost-5hf5)  
88. AWS Lambda Python with Agent Squad, accessed May 21, 2025, [https://awslabs.github.io/agent-squad/cookbook/lambda/aws-lambda-python/](https://awslabs.github.io/agent-squad/cookbook/lambda/aws-lambda-python/)  
89. How to Integrate OpenAI with Google Cloud Platform \- Omi AI, accessed May 21, 2025, [https://www.omi.me/blogs/ai-integrations/how-to-integrate-openai-with-google-cloud-platform](https://www.omi.me/blogs/ai-integrations/how-to-integrate-openai-with-google-cloud-platform)  
90. How to develop Python Google Cloud Functions \- TimVink.nl, accessed May 21, 2025, [https://timvink.nl/blog/google-cloud-functions/](https://timvink.nl/blog/google-cloud-functions/)  
91. Executing asynchronous tasks | Cloud Run Documentation \- Google Cloud, accessed May 21, 2025, [https://cloud.google.com/run/docs/triggering/using-tasks](https://cloud.google.com/run/docs/triggering/using-tasks)  
92. Compare Agent Development Kit (ADK) vs. Swarm in 2025 \- Slashdot, accessed May 21, 2025, [https://slashdot.org/software/comparison/Agent-Development-Kit-ADK-vs-OpenAI-Swarm/](https://slashdot.org/software/comparison/Agent-Development-Kit-ADK-vs-OpenAI-Swarm/)  
93. Quickstart: Develop and deploy agents on Vertex AI Agent Engine \- Google Cloud, accessed May 21, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/quickstart](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/quickstart)  
94. Vertex AI Agent Engine overview | Generative AI on Vertex AI ..., accessed May 21, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview)  
95. Recap of Interrupt 2025: The AI Agent Conference by LangChain, accessed May 21, 2025, [https://blog.langchain.dev/interrupt-2025-recap/](https://blog.langchain.dev/interrupt-2025-recap/)  
96. How to think about agent frameworks \- LangChain Blog, accessed May 21, 2025, [https://blog.langchain.dev/how-to-think-about-agent-frameworks/](https://blog.langchain.dev/how-to-think-about-agent-frameworks/)  
97. Installation \- Swarms Docs, accessed May 21, 2025, [https://docs.swarms.world/en/latest/swarms/install/install/](https://docs.swarms.world/en/latest/swarms/install/install/)  
98. Pipelines and workflows \- OpenAI Agents SDK, accessed May 21, 2025, [https://openai.github.io/openai-agents-python/voice/pipeline/](https://openai.github.io/openai-agents-python/voice/pipeline/)  
99. How docs AI search works: Mintlify-Style with OpenAI Agents SDK \- DEV Community, accessed May 21, 2025, [https://dev.to/siddhantkcode/how-docs-ai-search-works-mintlify-style-with-openai-agents-sdk-121j](https://dev.to/siddhantkcode/how-docs-ai-search-works-mintlify-style-with-openai-agents-sdk-121j)  
100. Performance issues in a multi-agent scenario \- API \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/performance-issues-in-a-multi-agent-scenario/1129289](https://community.openai.com/t/performance-issues-in-a-multi-agent-scenario/1129289)  
101. LangChain \+ LangGraph vs. OpenAI Swarm \+ GPTs \+ Function Calling: A Comparison for Web Service Development \- NinjaLABO, accessed May 21, 2025, [https://ninjalabo.ai/blogs/lang\_vs\_swarm.html](https://ninjalabo.ai/blogs/lang_vs_swarm.html)  
102. LangChain, accessed May 21, 2025, [https://www.langchain.com/](https://www.langchain.com/)  
103. langchain-ai/langgraph: Build resilient language agents as graphs. \- GitHub, accessed May 21, 2025, [https://github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)  
104. LangChain, LangGraph Open Tutorial for everyone\! \- GitHub, accessed May 21, 2025, [https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial](https://github.com/LangChain-OpenTutorial/LangChain-OpenTutorial)  
105. LangGraph vs CrewAI vs OpenAI Swarm: Which AI Agent Framework to Choose? \- Oyelabs, accessed May 21, 2025, [https://oyelabs.com/langgraph-vs-crewai-vs-openai-swarm-ai-agent-framework/](https://oyelabs.com/langgraph-vs-crewai-vs-openai-swarm-ai-agent-framework/)  
106. Installation \- CrewAI, accessed May 21, 2025, [https://docs.crewai.com/installation](https://docs.crewai.com/installation)  
107. CrewAI \+ Groq: High-Speed Agent Orchestration \- GroqDocs, accessed May 21, 2025, [https://console.groq.com/docs/crewai](https://console.groq.com/docs/crewai)  
108. Building a Github repo summarizer with CrewAI | crewai\_git\_documenter \- Wandb, accessed May 21, 2025, [https://wandb.ai/byyoung3/crewai\_git\_documenter/reports/Building-a-Github-repo-summarizer-with-CrewAI--VmlldzoxMjY5Mzc5Ng](https://wandb.ai/byyoung3/crewai_git_documenter/reports/Building-a-Github-repo-summarizer-with-CrewAI--VmlldzoxMjY5Mzc5Ng)  
109. Github Search \- CrewAI, accessed May 21, 2025, [https://docs.crewai.com/tools/githubsearchtool](https://docs.crewai.com/tools/githubsearchtool)  
110. Handoffs — AutoGen \- Microsoft Open Source, accessed May 21, 2025, [https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/handoffs.html](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/handoffs.html)  
111. Develop an Agent Development Kit agent | Generative AI on Vertex AI \- Google Cloud, accessed May 21, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/adk](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/adk)  
112. google/adk-java \- GitHub, accessed May 21, 2025, [https://github.com/google/adk-java](https://github.com/google/adk-java)  
113. Google AI Agent ADK versus Chatgpts SDK : r/AI\_Agents \- Reddit, accessed May 21, 2025, [https://www.reddit.com/r/AI\_Agents/comments/1kopl1a/google\_ai\_agent\_adk\_versus\_chatgpts\_sdk/](https://www.reddit.com/r/AI_Agents/comments/1kopl1a/google_ai_agent_adk_versus_chatgpts_sdk/)  
114. Exploring Google's Agent Development Kit \- GetStream.io, accessed May 21, 2025, [https://getstream.io/blog/exploring-google-adk/](https://getstream.io/blog/exploring-google-adk/)  
115. Is Google Agent Development Kit (ADK) really worth the hype ? : r/AI\_Agents \- Reddit, accessed May 21, 2025, [https://www.reddit.com/r/AI\_Agents/comments/1k4erny/is\_google\_agent\_development\_kit\_adk\_really\_worth/](https://www.reddit.com/r/AI_Agents/comments/1k4erny/is_google_agent_development_kit_adk_really_worth/)  
116. Agent Development Kit: Making it easy to build multi-agent applications, accessed May 21, 2025, [https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)  
117. Google Introduces Agent Development Kit (ADK) at Cloud NEXT 2025 \- Blogs, accessed May 21, 2025, [https://blogs.infoservices.com/google-cloud/smart-ai-agents-google-agent-development-kit/](https://blogs.infoservices.com/google-cloud/smart-ai-agents-google-agent-development-kit/)  
118. Test the Trustworthiness of Agents built with Google ADK \- Vijil, accessed May 21, 2025, [https://www.vijil.ai/blog/test-the-trustworthiness-of-agents-built-with-google-adk](https://www.vijil.ai/blog/test-the-trustworthiness-of-agents-built-with-google-adk)  
119. We need typescript agent sdk \- API \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/we-need-typescript-agent-sdk/1144653](https://community.openai.com/t/we-need-typescript-agent-sdk/1144653)  
120. SentientGPT: A research project tackling AI's biggest weakness—memory loss between sessions \- Use cases and examples \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/sentientgpt-a-research-project-tackling-ai-s-biggest-weakness-memory-loss-between-sessions/1116179](https://community.openai.com/t/sentientgpt-a-research-project-tackling-ai-s-biggest-weakness-memory-loss-between-sessions/1116179)  
121. Agents SDK compatibility: Realtime preview model or Azure OpenAI endpoints? \- API, accessed May 21, 2025, [https://community.openai.com/t/agents-sdk-compatibility-realtime-preview-model-or-azure-openai-endpoints/1142648](https://community.openai.com/t/agents-sdk-compatibility-realtime-preview-model-or-azure-openai-endpoints/1142648)  
122. Agents SDK with Azure hosted models \- API \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/agents-sdk-with-azure-hosted-models/1157781](https://community.openai.com/t/agents-sdk-with-azure-hosted-models/1157781)  
123. Support for LiteLLM Tools (e.g., {"googleSearch": {}}) in OpenAI Agents SDK \- API, accessed May 21, 2025, [https://community.openai.com/t/support-for-litellm-tools-e-g-googlesearch-in-openai-agents-sdk/1240585](https://community.openai.com/t/support-for-litellm-tools-e-g-googlesearch-in-openai-agents-sdk/1240585)  
124. OpenAI Agents SDK \+ Multiple MCP Servers \- DEV Community, accessed May 21, 2025, [https://dev.to/seratch/openai-agents-sdk-multiple-mcp-servers-8d2](https://dev.to/seratch/openai-agents-sdk-multiple-mcp-servers-8d2)  
125. Topics tagged swarm \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/tag/swarm](https://community.openai.com/tag/swarm)  
126. Agents SDK : Handoffs \+ Tools across multiple agents \- API \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/agents-sdk-handoffs-tools-across-multiple-agents/1213811](https://community.openai.com/t/agents-sdk-handoffs-tools-across-multiple-agents/1213811)  
127. Getting Agent events with agents-as-tools \- API \- OpenAI Developer Community, accessed May 21, 2025, [https://community.openai.com/t/getting-agent-events-with-agents-as-tools/1165936](https://community.openai.com/t/getting-agent-events-with-agents-as-tools/1165936)