Forging Intelligent Assistants: A Developer's Guide to
Google's AI Agent Ecosystem (May 2025)

1. The Rise of AI Agents and Google's Vision

Artificial intelligence (AI) agents, sophisticated software entities capable of perceiving
their environment, making decisions, and taking autonomous actions to achieve
specific goals, are rapidly transitioning from conceptual frameworks to practical,
high-impact applications across diverse industries.1 Their ability to automate complex
tasks, personalize user experiences, and drive operational efficiencies has positioned
them as a transformative technology. Recognizing this potential, Google has
embarked on a significant strategic initiative to build a comprehensive and accessible
ecosystem for AI agent development and deployment. This commitment is evidenced
by a wave of product announcements and enhancements unveiled at prominent
events such as Google Cloud Next '25 and Google I/O 2025.3

The sheer volume and scope of these releases—spanning development kits, tool
integrations, deployment engines, and enterprise-facing platforms—underscore an
aggressive push by Google into the agentic AI landscape. This concerted effort across
multiple product lines points to a top-level strategic directive aimed at establishing a
leading position in this emerging market. The overarching goal appears to be the
democratization of AI agent development and usage, empowering both individual
developers and large enterprises to harness the power of intelligent automation.5 This
focus is particularly apparent in the design of tools that aim to simplify the creation of
agents capable of handling complex, multi-step, and cross-organizational workflows
without constant human intervention.6 For developers and organizations investing in
this ecosystem, this signals a strong likelihood of continued support, innovation, and a
rich feature roadmap.

Why Python for Google AI Agents?

Python's deep-rooted dominance in the AI and machine learning (ML) landscape
makes it a natural and strategic choice for Google's agent development tools. Its
extensive libraries, mature ecosystem, and developer-friendly syntax have made it the
de facto language for AI research and production systems. Google has embraced this
by centering its primary agent development framework, the Agent Development Kit
(ADK), around Python.4

As of May 2025, the Python ADK has reached version 1.0.0, a milestone signifying its
stability and readiness for production environments.4 This maturity is crucial for
developers looking to build and deploy robust, enterprise-grade agents. Furthermore,

Google provides dedicated Python Software Development Kits (SDKs) for various
critical components within its agent ecosystem, including the Agent-to-Agent (A2A)
communication protocol and the Vertex AI Agent Engine, facilitating seamless
integration and development workflows.4

While Python is clearly the primary language for ADK development at this juncture,
the introduction of an initial release of the Java ADK (v0.1.0) in May 2025 is a
noteworthy development.3 This signals Google's intent to broaden the appeal and
accessibility of its agent-building tools to other large enterprise developer
communities where Java has a strong foothold. This phased approach, prioritizing the
dominant AI language first, suggests a long-term vision for a more inclusive,
multi-language agent development environment.

Navigating This Guide: Key Technologies Covered

This guide provides a comprehensive exploration of Google's AI agent ecosystem,
focusing on Python-based development and the features available as of May 2025. It
will delve into the following core Google technologies, offering practical explanations
and code examples:

●  Agent Development Kit (ADK): The foundational framework for building agents.
●  Agent Tools: Essential integrations including Retrieval Augmented Generation
(RAG) engines, Google Search, Vertex AI Search, code execution capabilities,
connectors for over 100 enterprise applications, and support for popular
open-source frameworks like LangChain and CrewAI.

●  Agent Memory: Mechanisms for managing short-term session context and

long-term knowledge.

●  Model Context Protocol (MCP): An open standard for tool communication.
●  Agent-to-Agent (A2A) Communication: A protocol for enabling collaboration

between agents.

●  Vertex AI Agent Engine: The managed runtime for deploying and scaling agents

in production.

●  Vertex AI Agent Builder: The overarching suite of tools for agent development.
●  AgentSpace: The enterprise platform for discovering and utilizing AI agents,

including a look at banner prebuilt agents.

Insights & Implications for this Section

The rapid succession of releases and enhancements across Google's agent stack,
highlighted at major industry events, points to a significant strategic investment.
Google is clearly aiming to provide a comprehensive, end-to-end platform for agentic
AI, from development to deployment and enterprise-wide adoption. This aggressive

strategy suggests that the ecosystem will continue to evolve rapidly, offering
increasingly powerful tools and capabilities.

The choice of Python as the initial primary language for the ADK aligns with its
prevalence in the AI/ML community, ensuring a large pool of developers can readily
adopt the framework. However, the concurrent introduction of a Java ADK, albeit in an
earlier version, indicates a broader strategy to engage diverse enterprise development
teams. This suggests that while Python currently leads, the ecosystem is being built
with future language expansion in mind, potentially lowering barriers to entry for
organizations with varied technology stacks.

2. The Foundation: Google's Agent Development Kit (ADK)

Understanding ADK: An Open, Modular Framework

The Google Agent Development Kit (ADK) serves as the cornerstone for developers
looking to build, manage, evaluate, and deploy sophisticated AI-powered agents
within Google's ecosystem.1 It is an open-source framework meticulously designed to
simplify the often-complex process of creating multi-agent systems while affording
developers precise control over agent behavior and orchestration.10

A key characteristic of the ADK is its flexibility and modularity. While optimized for
seamless integration with Google's Gemini family of models and the broader Google
Cloud ecosystem, the ADK is fundamentally model-agnostic and
deployment-agnostic.3 This means developers are not strictly locked into Google's
offerings and can potentially integrate other large language models (LLMs) or deploy
agents to custom environments. The framework's code-first approach, particularly in
Python, resonates with developers who prefer programmatic control and the ability to
integrate agent development into existing software engineering practices like version
control and automated testing.14 This developer-centric philosophy is evident in its
design, aiming to make agent development feel more akin to traditional software
development.14 This approach contrasts with more opaque or purely UI-driven agent
builders, offering a higher degree of customizability and transparency, which is often
crucial for complex enterprise applications.

Python ADK v1.0.0: Production-Ready Agent Development

A significant milestone for the ADK was achieved in May 2025 with the release of
Python ADK v1.0.0.4 This stable release officially marks the Python ADK as
production-ready, providing a reliable and robust platform for developers to
confidently build and deploy their agents in live environments. The maturity and
stability of this version are underscored by early adoption and positive feedback from

prominent companies such as Renault Group, Box, and Revionics, who are already
leveraging the ADK for their agent development needs.4

Concurrently, Google also launched the initial release of the Java ADK v0.1.0.3 While
the Python ADK is more mature, the introduction of the Java ADK signals Google's
commitment to extending the power and flexibility of the ADK to the vast Java
developer community. This phased maturation and ecosystem building—stabilizing
the Python version while introducing Java support and simultaneously expanding tool
integrations and deployment options—indicates a deliberate strategy. This approach
allows Google to incorporate developer feedback and ensure robustness before
pushing for broader enterprise adoption across different technology stacks.

Setting Up Your Python ADK Development Environment

To begin developing AI agents with the Python ADK, a proper development
environment setup is essential. This typically involves creating an isolated Python
environment and installing the necessary ADK packages.

1.  Create and Activate a Virtual Environment (Recommended):

It is highly recommended to use a virtual environment to manage project
dependencies and avoid conflicts.
Bash
# Create a virtual environment (e.g., named.venv)
python -m venv.venv

# Activate the virtual environment
# On macOS/Linux:
source.venv/bin/activate
# On Windows CMD:
#.venv\Scripts\activate.bat
# On Windows PowerShell:
#.venv\Scripts\Activate.ps1

9

2.  Install the Agent Development Kit:

Once the virtual environment is activated, the ADK can be installed using pip:
Bash
pip install google-adk

9
This command installs the core ADK framework dependency. The ADK also comes
with an integrated web server that provides a development user interface (Dev UI)

for interacting with and debugging agents locally.3

3.  Configure Environment Variables:

ADK agents, particularly those interacting with LLMs like Gemini, require API keys
and configuration settings passed via environment variables. Create a .env file in
your project's agent directory (e.g., multi_tool_agent/.env) with the following:
Code snippet
GOOGLE_API_KEY=PASTE_YOUR_GEMINI_API_KEY_HERE
GOOGLE_GENAI_USE_VERTEXAI=FALSE

3

○  GOOGLE_API_KEY: Your API key for Gemini, obtainable from Google AI Studio.
○  GOOGLE_GENAI_USE_VERTEXAI: Set to FALSE if using the Gemini API directly
(e.g., via AI Studio key). If set to TRUE, the agent will attempt to use Vertex AI
for model inference, requiring appropriate Google Cloud project setup and
authentication. For Vertex AI usage, additional variables like
GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION would be
needed in the .env file.9

Crafting Your First Python ADK Agent (with code examples)

With the environment set up, one can proceed to create a basic Python ADK agent.
This involves defining the agent's structure, its core logic (instructions), and any tools
it can utilize.

1.  Project Structure:

A typical ADK agent project might have the following structure 9:
parent_folder/
└── multi_tool_agent/
    ├── __init__.py
    ├── agent.py
    └──.env

○  multi_tool_agent/: The root directory for this specific agent.
○  __init__.py: Makes the directory a Python package and typically imports the

agent module.

○  agent.py: Contains the core agent definition and its associated tools.
○

.env: Stores environment variables like API keys.

2.  __init__.py:

This file makes the agent module accessible.
Python
# multi_tool_agent/__init__.py

from. import agent

9

3.  agent.py (Example: Weather and Time Agent):

This file defines the agent's behavior, the LLM it uses, and its tools.
Python
# multi_tool_agent/agent.py
import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

# Define a Python function as a tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.
    Args:
        city (str): The name of the city for which to retrieve the weather report.
    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

# Define another Python function as a tool
def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.
    Args:
        city (str): The name of the city for which to retrieve the current time.
    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":

        tz_identifier = "America/New_York"
    elif city.lower() == "london":
        tz_identifier = "Europe/London"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }
    try:
        tz = ZoneInfo(tz_identifier)
        now = datetime.datetime.now(tz)
        report = (
            f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
        )
        return {"status": "success", "report": report}
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Could not retrieve time for {city}: {str(e)}",
        }

# Define the root agent
root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",  # Specify the LLM to use
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time "
        "and weather in a city. Use the available tools to find the information. "
        "If a city is not supported by a tool, inform the user politely."
    ),
    tools=[get_weather, get_current_time],  # Register the function tools
)

9

In this example, Agent is instantiated with a name, the chosen model (e.g.,
gemini-2.0-flash), a description of its purpose, a system instruction guiding its
behavior and tool usage, and a list of tools it can employ. The Python functions
get_weather and get_current_time are directly passed as tools. The ADK
framework will introspect these functions (including their docstrings and type
hints) to make them callable by the LLM.

4.  .env File Configuration:

As mentioned in the setup, the .env file in the multi_tool_agent/ directory should
contain:
Code snippet
# multi_tool_agent/.env
GOOGLE_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
GOOGLE_GENAI_USE_VERTEXAI=FALSE

9
Replace YOUR_ACTUAL_GEMINI_API_KEY with the key obtained from Google AI
Studio.

Leveraging the ADK Dev UI for Iteration and Debugging

The ADK includes a powerful Development User Interface (Dev UI) that significantly
aids in testing, iterating, and debugging agents locally.3

●  Launching the Dev UI:

Navigate to the parent directory of your agent project (e.g., parent_folder/) in the
terminal and run:
Bash
adk web

●

9
This command starts a local web server, typically accessible at
http://localhost:8000.
Interacting with Your Agent:
In the Dev UI, you can select your agent (e.g., "multi_tool_agent") from a
dropdown. A chat interface allows you to send messages to the agent and
observe its responses. The UI also displays an "Events" tab, which provides a
detailed trace of the agent's execution, including LLM calls, tool selections, tool
inputs/outputs, and final responses. This granular view is invaluable for
understanding agent behavior and debugging issues.3 The InstaVibe Codelab
further demonstrates the utility of the Dev UI, including an "Eval" tab for
systematic testing with predefined inputs and evaluation criteria.15

●  Alternative Running Methods:

Besides the Dev UI, ADK offers other ways to run and interact with agents 9:
○  Terminal (adk run): For direct command-line interaction:

Bash
adk run multi_tool_agent

○  API Server (adk api_server): To expose the agent as a local FastAPI server,

useful for testing API integrations before deployment.

Key ADK Concepts

The ADK framework is built upon several core concepts that manage the agent's
lifecycle and conversational context. These are well-illustrated in practical examples
like the InstaVibe Codelab 15:

●  Session: A Session acts as a container for a single, coherent interaction or chat
thread with an agent. It holds the conversation history, the current working data
(State), and associated metadata.

●  State: The State object represents the agent's short-term, mutable working

memory within a specific session. It's used to store temporary information that
might be needed across multiple turns of a conversation or steps in a workflow.

●  Memory: Distinct from State, Memory (often managed by a MemoryService)

refers to capabilities for long-term recall, potentially across different sessions or
by accessing external knowledge bases. This is covered in detail in Section 4.
●  Event: An Event is an immutable record of every significant interaction or action
that occurs within a session. This includes user messages, agent responses, tool
calls, and tool returns, forming a chronological log that is crucial for debugging
and understanding the agent's reasoning process.

●  Runner: The Runner is the core execution engine within ADK. It orchestrates the
agent's operation by managing sessions, processing events, updating state,
invoking the underlying LLM, and coordinating tool calls.

Understanding these components is fundamental to building more complex and
stateful agents with ADK. The developer-centric design of ADK, with its emphasis on
code-first development, local iteration via the Dev UI, and an open-source foundation,
positions it as an attractive framework for developers seeking fine-grained control
and seamless integration with established software development practices.

Table 2.1: ADK Core Components Overview

Component

Description

Agent

Runner

Session

State

Event

Tool

Dev UI

Key Python Class/Usage
(Conceptual)

from google.adk.agents
import Agent <br> my_agent =
Agent(...)

The fundamental unit
representing an AI assistant,
defining its model,
instructions, tools, and
behavior.

The core execution engine
that manages agent lifecycles,
sessions, events, and
LLM/tool invocations.

from google.adk.runners
import Runner <br> runner =
Runner(agent=my_agent,...)
<br> runner.run(...)

A container for a single chat
thread or interaction, holding
history, state, and metadata.

Managed by SessionService
(e.g.,
InMemorySessionService)

The agent's short-term,
mutable working memory
within a session, storing
temporary contextual
information.

Accessed via
tool_context.state or within
agent logic (e.g., in
LoopAgent callbacks).

An immutable record of every
interaction (user input, agent
response, tool call/return)
within a session.

Iterated over in runner.run(...)
output.

A specific capability (e.g.,
Python function, built-in
service) given to an agent to
perform actions.

Passed in
tools=[my_function_tool] list
during Agent instantiation.

A web-based interface for
local development, testing,
and debugging of ADK
agents.

Launched via adk web
command.

This table provides a quick reference to the fundamental building blocks of ADK,
helping developers grasp the main concepts and their corresponding Python
implementations, thereby accelerating the learning process.3

3. Equipping Your Agents: Mastering Tool Integration

The Power of Tools in ADK Agents

In the Agent Development Kit (ADK), tools are fundamental extensions that bestow
agents with capabilities far exceeding the inherent reasoning and text generation
functions of their underlying Large Language Models (LLMs).17 These tools empower
agents to interact with the external world, access real-time data, execute specific
actions, and connect with a multitude of systems. Essentially, tools transform an LLM
from a sophisticated text processor into a functional agent capable of performing
tasks.

The ADK framework facilitates a structured process for how agents utilize tools 17:

1.  Reasoning & Selection: Based on the user's query, conversation history, and its
own instructions, the agent's LLM reasons about the task at hand. It then selects
the most appropriate tool(s) from its registered arsenal. The clarity and
descriptiveness of a tool's function signature and docstring are crucial for the
LLM to make an accurate selection.

2.  Invocation: The LLM generates the necessary arguments for the selected tool

and triggers its execution.

3.  Observation: The agent receives the output or result returned by the executed

tool.

4.  Finalization: The agent incorporates the tool's output into its ongoing reasoning
process to formulate the next response, decide subsequent actions, or determine
if the goal has been achieved.

ADK supports a diverse range of tool types, enabling developers to equip their agents
with a wide spectrum of functionalities. These include custom Python Function tools,
pre-packaged Built-in tools, integrations with popular Third-party libraries,
connections to Google Cloud services, and tools adhering to the Model Context
Protocol (MCP).13 This rich tool ecosystem is a testament to Google's strategy of
fostering openness and extensibility, allowing developers to leverage existing assets
and skills rather than being confined to a proprietary, limited set of capabilities. This
approach accelerates development and broadens the applicability of ADK agents.

Core Built-in Tools

ADK provides several built-in tools for common functionalities, which can be easily
integrated into agents by importing, configuring (if necessary), and registering them in
the agent's tool list.18

Grounding with Google Search

●  Purpose: This tool enables ADK agents to access and retrieve real-time

information from the public web using Google Search.13 This is invaluable for tasks
requiring up-to-date knowledge that may not be present in the LLM's training
data.

●  Python Usage Example:

To use the Google Search tool, import google_search from google.adk.tools and
include it in the tools list when defining an Agent. The agent's instruction should
guide it on when and how to utilize this search capability. The InstaVibe Codelab
provides a practical example where a planner agent uses Google Search to find
current events and venues.15
Python
from google.adk.agents import Agent
from google.adk.tools import google_search # Import the built-in Google Search tool

# Define an agent that can use Google Search
web_search_agent = Agent(
    name="web_researcher_agent",
    model="gemini-2.0-flash",
    description="An agent that can search the web for current information.",
    instruction=(
        "You are a helpful assistant. If the user asks for information "
        "that is likely to be recent or requires real-time data, "
        "use the google_search tool to find the answer. "
        "Clearly state that you are searching the web."
    ),
    tools=[google_search] # Add the google_search tool to the agent's capabilities
)

# To run this agent (conceptual, assuming runner and session setup as in Section 2):
# user_query = "What are the latest developments in AI agent technology as of May 2025?"
#... run agent with user_query...

15

Enterprise Knowledge with Vertex AI Search

●  Purpose: The Vertex AI Search tool allows ADK agents to query and retrieve

information from private enterprise data sources that have been indexed within
Vertex AI Search applications or datastores.13 This is crucial for building agents

that can provide contextually relevant answers based on an organization's internal
knowledge. Grounding on data using Vertex AI Search became generally available
in May 2025.19

●  Python Usage (Conceptual): While specific ADK code examples for directly
using a VertexAiSearchTool are less explicit in the provided materials than for
google_search, it is listed as a built-in tool.13 Integration would involve registering
the tool with the agent. The agent's instructions would then guide it to query
specific enterprise knowledge when appropriate. Configuration would likely
involve specifying the relevant Vertex AI Search application ID or datastore ID,
potentially during tool initialization or through environment variables. ADK's
general support for grounding with Vertex AI Search is also noted.10

Dynamic Capabilities with Code Execution

●  Purpose: The Code Execution tool provides ADK agents with the ability to
generate and execute arbitrary Python code snippets dynamically.13 This is
extremely powerful for tasks involving calculations, data manipulation, algorithmic
logic, or any operation that can be expressed in Python code but is not suitable
for a pre-defined function tool.

●  Python Usage (Conceptual): As a built-in tool 13, the agent would be instructed
by the developer to generate Python code to solve a particular sub-problem and
then use the Code Execution tool to run that code. The LLM would need to be
carefully prompted to produce safe and correct code. The tool would then return
the execution result (or error) to the agent for further processing.

Advanced Retrieval with Vertex AI RAG Engine

Retrieval Augmented Generation (RAG) is a critical technique for enhancing LLM
responses by connecting them to external, up-to-date, and often proprietary
knowledge sources. This helps to mitigate hallucinations, improve factual accuracy,
and provide more relevant, context-aware answers.20 Google's Vertex AI RAG Engine is
a managed orchestration service designed to simplify the implementation of RAG
pipelines.20 The prominent placement of RAG Engine as an agent tool 13 and the
availability of a dedicated service underscore the importance Google places on
grounding for building reliable enterprise agents.

●

Implementing RAG with Vertex AI RAG Engine:
The Vertex AI SDK for Python provides the necessary functionalities to set up and
use the RAG Engine.
1.  Corpus Setup: A RAG Corpus is a collection of documents that the agent can

search. It's created using rag.create_corpus(), specifying an embedding

model configuration.21
Python
from vertexai import rag
import vertexai

# Assuming vertexai.init() has been called with project and location
# PROJECT_ID = "your-project-id"
# vertexai.init(project=PROJECT_ID, location="us-central1")

embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005" # Example
embedding model
    )
)
rag_corpus = rag.create_corpus(
    display_name="my_knowledge_corpus",
    backend_config=rag.RagVectorDbConfig(
        rag_embedding_model_config=embedding_model_config
    ),
)
print(f"Created RAG Corpus: {rag_corpus.name}")

21

2.  Data Ingestion: Documents (from Google Cloud Storage or Google Drive) are

imported into the corpus using rag.import_files(). This process involves
chunking the documents into manageable pieces for embedding and
retrieval.21
Python
# paths = ["gs://your-bucket/your-document.pdf"] # Example GCS path
# rag.import_files(
#     rag_corpus.name,
#     paths,
#     transformation_config=rag.TransformationConfig(
#         chunking_config=rag.ChunkingConfig(
#             chunk_size=512,
#             chunk_overlap=100,
#         ),
#     ),
# )
# print(f"Files imported into {rag_corpus.name}")

21

3.  Retrieval Queries: To fetch relevant context, rag.retrieval_query() is used,

specifying the RagResource (corpus) and retrieval configuration (e.g., top_k
results).21
Python
# rag_retrieval_config = rag.RagRetrievalConfig(top_k=3)
# response = rag.retrieval_query(
#     rag_resources=,
#     text="What is the main topic of the ingested documents?",
#     rag_retrieval_config=rag_retrieval_config,
# )
# print("Retrieved contexts:", response)

21

●

Integrating RAG Tools with ADK Agents and Gemini Models:
The retrieved context from the RAG Engine can then be used to augment the
prompt for a generative model like Gemini, enabling it to produce more informed
responses. This is facilitated by creating a RAG retrieval tool.
1.  Creating a RAG Retrieval Tool for Gemini:

The Tool.from_retrieval() method, using rag.VertexRagStore, creates a tool
compatible with Gemini models.21
Python
from vertexai.generative_models import GenerativeModel, Tool

# Assuming rag_corpus and rag_retrieval_config are defined as above
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=,
            rag_retrieval_config=rag_retrieval_config, # Defined earlier
        ),
    )
)

21

2.  Adding the RAG Tool to an ADK Agent (via Gemini Model):

An ADK agent that uses a Gemini model can leverage this RAG tool by
including it in the model's tool configuration.
Python
# Define a Gemini model instance with the RAG tool
# rag_model_for_agent = GenerativeModel(
#     model_name="gemini-2.0-flash-001", # Or other compatible Gemini model

#     tools=[rag_retrieval_tool]
# )

# Conceptually, an ADK agent would then use this configured model:
# from google.adk.agents import Agent
# knowledge_agent = Agent(
#     name="knowledge_retriever_agent",
#     model=rag_model_for_agent, # Pass the Gemini model configured with the RAG tool
#     description="An agent that answers questions based on a knowledge corpus.",
#     instruction=(
#         "You are an expert Q&A assistant. Use the RAG retrieval tool "
#         "to find relevant information from the knowledge corpus before answering."
#     )
#     # Note: The ADK's direct `model` parameter might take a string like "gemini-2.0-flash".
#     # The exact mechanism for passing a pre-configured GenerativeModel instance with
tools
#     # to an ADK Agent needs to be verified with the latest ADK documentation.
#     # However, the principle is that the Gemini model used by the ADK agent is made
RAG-aware.
# )

# Example generation using the RAG-enabled model directly:
# response = rag_model_for_agent.generate_content(
#     "Based on the documents, what are the key challenges?"
# )
# print("Generated response with RAG:", response.text)

21
The ADK agent's instructions would then guide it to use this RAG retrieval tool
when faced with queries requiring information from the specific knowledge
corpus. The GitHub repository adk-vertex-ai-rag-engine further illustrates a
complete workflow for setting up GCS buckets, uploading documents,
creating RAG corpora, and querying them, providing a practical example of
these steps in action.22

Bridging to Enterprise Systems: Google Cloud Tools

A significant aspect of building impactful AI agents is enabling them to interact with
existing enterprise systems and workflows. Google Cloud provides robust tools for
this, primarily through Application Integration and Integration Connectors, which are
designed to connect agents to a vast array of business applications. This focus on
enterprise connectivity underscores Google's strategy to empower agents to perform
meaningful, value-added tasks within complex organizational environments.

Application Integration & Integration Connectors (100+ Apps)

●  Purpose: These tools provide a secure and governed way for ADK agents to

connect to over 100 pre-built connectors for enterprise applications (such as
Salesforce, ServiceNow, JIRA, SAP) and to trigger custom integration workflows
built with Google Cloud Application Integration.13 This capability is essential for
automating business processes that span multiple systems.

●  Using ApplicationIntegrationToolset:

The ApplicationIntegrationToolset from
google.adk.tools.application_integration_tool.application_integration_toolset is
the key Python class for this integration.23

●  Prerequisites 23:

○  An installed and configured Google Cloud CLI.
○  An existing Application Integration workflow or Integration Connectors

connection.

○  For connectors, Application Integration must be provisioned in the same

region as the connection, and the ExecuteConnection template integration
must be created and published.
●  Python Example (Integration Connectors):

To use a specific connector, instantiate ApplicationIntegrationToolset with details
of the GCP project, location, connection name, and the desired entity operations
or actions. The get_tools() method then makes these available to the ADK agent.
Python
# In your agent's tools.py
from google.adk.tools.application_integration_tool.application_integration_toolset
import ApplicationIntegrationToolset

# Example: Connecting to a Salesforce connector
salesforce_connector_tool = ApplicationIntegrationToolset(
    project="your-gcp-project-id",    # Replace with your GCP project ID
    location="us-central1",           # Replace with the connection's region
    connection="my-salesforce-connection", # Replace with your Salesforce connection name
    # Define entities and operations the agent can use, e.g., list Accounts, create Leads
    entity_operations={"Account":, "Lead":},
    # Optional: service_account_credentials='{...}' if not using default gcloud auth
    tool_name="SalesforceTools", # Prefix for tool names generated
    tool_instructions="Use these tools to interact with Salesforce."
)

# In your agent.py

# from.tools import salesforce_connector_tool
# from google.adk.agents import LlmAgent
#
# enterprise_agent = LlmAgent(
#     model='gemini-2.0-flash',
#     name='salesforce_data_agent',
#     instruction="Interact with Salesforce to retrieve or update customer data as requested.",
#     tools=salesforce_connector_tool.get_tools(), # Make connector tools available
# )

23

●  Python Example (Application Integration Workflows):

To trigger an existing Application Integration workflow, instantiate
ApplicationIntegrationToolset with the project, location, integration name, and
trigger ID.
Python
# In your agent's tools.py
from google.adk.tools.application_integration_tool.application_integration_toolset
import ApplicationIntegrationToolset

# Example: Triggering an order processing workflow
order_processing_tool = ApplicationIntegrationToolset(
    project="your-gcp-project-id",    # Replace with your GCP project ID
    location="us-central1",           # Replace with the integration's region
    integration="process-new-order-workflow", # Replace with your integration name
    trigger="api_trigger/startOrderProcessing", # Replace with your API trigger ID
    tool_name="OrderWorkflow",
    tool_instructions="Use this tool to initiate the order processing workflow."
)

# In your agent.py (similar to the connector example, adding integration_tool.get_tools())
# from.tools import order_processing_tool
#...
# workflow_agent = LlmAgent(
#    ...
#     tools=order_processing_tool.get_tools(),
# )

23

Connecting to APIs via Apigee API Hub (Conceptual)

●  Purpose: The Apigee API Hub allows organizations to manage and catalog their

●

APIs. ADK agents can potentially leverage this by discovering and interacting with
these managed APIs.13
Integration Approach: While direct Python examples for an "ApigeeTool" are not
explicitly provided in the snippets, integration would likely occur through ADK's
support for OpenAPI tools (mentioned in ADK documentation navigation 11) if the
Apigee-managed APIs have OpenAPI specifications. Alternatively, custom
function tools could be written in Python to wrap calls to these APIs.

Extending with Open Source: LangChain and CrewAI

The ADK's design philosophy emphasizes extensibility and compatibility with the
broader AI ecosystem.14 This is clearly demonstrated by its support for integrating
tools from popular open-source agent frameworks like LangChain and CrewAI.

Integrating LangChain Tools into ADK

●  Mechanism: ADK provides the LangchainTool wrapper class, located in

google.adk.tools.langchain_tool, to seamlessly incorporate existing LangChain
tools into ADK agents.24

●  Python Example (Tavily Search Tool):

This example shows how to use LangChain's TavilySearchResults tool within an
ADK agent.
1.  Installation: pip install google-adk langchain langchain_community

tavily-python

2.  API Key: Set the TAVILY_API_KEY environment variable.
3.  Code:
Python
from google.adk.agents import Agent
from google.adk.tools.langchain_tool import LangchainTool
from langchain_community.tools import TavilySearchResults
import os

# Ensure TAVILY_API_KEY is set
# os.environ = "YOUR_TAVILY_API_KEY"

# 1. Instantiate the LangChain tool
tavily_search_instance = TavilySearchResults(
    max_results=3,
    search_depth="advanced",
    include_answer=True
)

# 2. Wrap it with LangchainTool for ADK
adk_tavily_tool = LangchainTool(tool=tavily_search_instance)

# 3. Add the wrapped tool to an ADK agent
langchain_powered_agent = Agent(
    name="langchain_tavily_agent",
    model="gemini-2.0-flash",
    description="An agent that uses LangChain's Tavily tool for web searches.",
    instruction="Use the Tavily search tool to answer questions about current events or
general knowledge.",
    tools=[adk_tavily_tool]
)

# Agent can now be run using ADK's Runner
24 It's also worth noting that the Vertex AI Agent Engine provides a
LangchainAgent template, which simplifies deploying LangChain-based
agents and supports LangChain tools.25

Integrating CrewAI Tools into ADK

●  Mechanism: Similarly, ADK offers the CrewaiTool wrapper from

google.adk.tools.crewai_tool for integrating tools from the CrewAI framework.24

●  Python Example (SerperDevTool for Web Search):

This example demonstrates using CrewAI's SerperDevTool for Google Search
results.
1.  Installation: pip install google-adk crewai-tools
2.  API Key: Set the SERPER_API_KEY environment variable.
3.  Code:
Python
from google.adk.agents import Agent
from google.adk.tools.crewai_tool import CrewaiTool
from crewai_tools import SerperDevTool
import os

# Ensure SERPER_API_KEY is set
# os.environ = "YOUR_SERPER_API_KEY"

# 1. Instantiate the CrewAI tool
serper_tool_instance = SerperDevTool(
    n_results=5,

    search_type="search" # "news", "images", etc.
)

# 2. Wrap it with CrewaiTool for ADK, providing a name and description for ADK's LLM
adk_serper_tool = CrewaiTool(
    name="InternetSearchNews", # Name used by ADK's LLM to identify the tool
    description="Searches the internet for general information or recent news articles using
Serper.",
    tool=serper_tool_instance
)

# 3. Add the wrapped tool to an ADK agent
crewai_powered_agent = Agent(
    name="crewai_serper_agent",
    model="gemini-2.0-flash",
    description="An agent that uses CrewAI's Serper tool for web searches.",
    instruction="Use the InternetSearchNews tool to find information online.",
    tools=[adk_serper_tool]
)

# Agent can now be run using ADK's Runner
24 A crucial aspect when wrapping CrewAI tools is providing explicit name and
description parameters to the CrewaiTool wrapper. These are used by ADK's
underlying LLM to understand the tool's purpose and when to invoke it.24

Standardized Tool Communication: Model Context Protocol (MCP)

The Model Context Protocol (MCP) is an open standard designed to streamline and
standardize how LLMs and AI agents communicate with external applications, data
sources, and tools.13 It defines a client-server architecture where an MCP server
exposes resources (data), interactive templates (prompts), and actionable functions
(tools) to MCP clients, which can be LLM host applications or AI agents like those built
with ADK.

ADK embraces MCP by supporting both the consumption of tools from existing MCP
servers and the exposure of ADK-native tools via a newly built MCP server.26 This dual
capability enhances interoperability and allows ADK agents to participate in broader,
standardized tool ecosystems.

●  Key MCP Integrations with ADK:

○  MCP Toolbox for Databases (GenAI Toolbox): An open-source MCP server

that enables agents to access data in databases. ADK has built-in support for
this toolbox.26

○  FastMCP: A Pythonic library that simplifies building MCP servers, often

requiring just a function decorator. ADK can integrate with FastMCP servers,
for instance, running on Cloud Run.26

○  MCP Tools for Genmedia Services: Open-source MCP servers for

integrating Google Cloud generative media services like Imagen (images), Veo
(video), Chirp (audio), and Lyria (music) into AI applications. ADK agents can
orchestrate these services via MCP.26

●  Python Usage for MCP Server Setup (InstaVibe Codelab Example):

The InstaVibe Codelab provides a practical demonstration of setting up an MCP
server in Python to expose an application's internal APIs as tools for ADK
agents.15
1.  API Wrapper Functions: Python functions are created to wrap the actual
HTTP calls to the application's APIs (e.g., create_post, create_event in
instavibe.py).15
Python
# Conceptual snippet from instavibe.py [15]
# def create_post_on_instavibe(author_name: str, text: str, sentiment: str):
#     #... logic to call InstaVibe's /posts API...
#     return response_json

2.  MCP Server Implementation (mcp_server.py):

An HTTP server (e.g., using FastAPI or a similar framework) implements MCP
endpoints:
■

list_tools: Allows MCP clients to discover available tools. It returns
metadata about each tool, including its name, description, and
input/output schema. The adk_to_mcp_tool_type utility can convert ADK
tool definitions to the MCP format.15

■  call_tool: Handles requests from clients to execute a specific tool. It

receives the tool name and arguments, invokes the corresponding Python
wrapper function, and returns the result.15

Python
# Conceptual MCP server endpoint structure [15]
# from fastmcp.server import app # Example if using FastMCP
# from google.adk.mcp import adk_to_mcp_tool_type, mcp_types
# from.instavibe_tools import instavibe_post_tool, instavibe_event_tool # ADK tools

# available_adk_tools = {
#     instavibe_post_tool.name: instavibe_post_tool,
#     instavibe_event_tool.name: instavibe_event_tool,
# }

# @app.list_tools() # Or equivalent for your chosen MCP server library
# async def list_mcp_tools() -> list:
#     mcp_tools =
#     for adk_tool_instance in available_adk_tools.values():
#         mcp_tools.append(adk_to_mcp_tool_type(adk_tool_instance))
#     return mcp_tools

# @app.call_tool() # Or equivalent
# async def call_mcp_tool(name: str, arguments: dict) -> list[mcp_types.Content]:
#     adk_tool_to_call = available_adk_tools.get(name)
#     if adk_tool_to_call:
#         adk_response = await adk_tool_to_call.run_async(args=arguments, tool_context=None)
#         # Convert adk_response to mcp_types.Content
#         response_text = json.dumps(adk_response)
#         return
#     else:
#         # Handle tool not found
#         return
This MCP server can then be deployed (e.g., to Cloud Run), and its URL can be used
by ADK agents configured to consume MCP tools.

Table 3.1: Agent Tool Integration Matrix

Tool Category

Key ADK Integration
Method/Wrapper

Primary Use Case

Example Snippet
Reference

Built-in: Google
Search

from google.adk.tools
import
google_search

Real-time web search
and information
retrieval.

Built-in: Vertex AI
Search

Built-in tool
(configuration
specific)

Searching over
private enterprise
data indexed in
Vertex AI Search.

Built-in: Code
Execution

Built-in tool (LLM
generates code to be
executed)

Dynamic calculations,
data manipulation,
custom logic
execution.

RAG: Vertex AI RAG
Engine

Tool.from_retrieval(re
trieval=rag.Retrieval(s
ource=rag.VertexRag
Store(...)))

Grounding agent
responses with
up-to-date, factual
enterprise

15

13

13

21

knowledge.

Google Cloud:
Connectors

ApplicationIntegratio
nToolset (for
connections)

Connecting to 100+
enterprise apps
(Salesforce, SAP,
etc.).

Google Cloud: App
Integration

ApplicationIntegratio
nToolset (for
integrations)

Triggering custom
enterprise workflows
built with Application
Integration.

LangChain Tools

from
google.adk.tools.lang
chain_tool import
LangchainTool

Integrating existing
tools from the
LangChain
framework.

CrewAI Tools

Model Context
Protocol (MCP)

from
google.adk.tools.crew
ai_tool import
CrewaiTool

Integrating existing
tools from the CrewAI
framework.

Python MCP server
implementation
(list_tools, call_tool
endpoints)

Standardized
communication with
external tools and
data sources via
MCP.

23

23

24

24

15

This matrix serves as a valuable quick-reference for developers, enabling them to
efficiently identify the appropriate ADK integration method for various tooling
requirements, thereby streamlining the development of capable and well-equipped AI
agents.

4. Orchestrating Conversations: Agent State and Memory
Management

The Critical Role of Memory in Conversational AI

For an AI agent to engage in coherent, personalized, and effective multi-turn
interactions, memory is indispensable.10 Without memory, each interaction would be
stateless, forcing the user to repeat information and preventing the agent from
learning preferences or recalling past context. Google's agent ecosystem, particularly
through the ADK and Vertex AI Agent Engine, provides mechanisms for both

short-term context management (session state) and long-term knowledge retention.10
This tiered approach to memory allows developers to choose the right solution based
on the complexity, persistence needs, and deployment environment of their agents.

Short-Term Context: Session State in ADK

●  Understanding the ADK State Object:

Within the Agent Development Kit, the State object serves as the primary
mechanism for managing short-term, mutable working memory within the scope
of a single agent session.11 This is akin to a scratchpad that the agent uses to
keep track of temporary information relevant to the current conversation or task.
For instance, if an agent is guiding a user through a multi-step process, the State
object can store the user's selections or intermediate results from previous steps.

●  Usage in Multi-Turn Dialogues and Workflows:

The State is crucial for maintaining context across multiple turns of a dialogue.
When an agent processes a user's message, it can read from and write to the
State object. This allows subsequent turns or tool executions within the same
session to access and build upon previously gathered information.
In more complex scenarios, such as ADK's LoopAgent (a type of workflow agent),
the State object is used to pass data between different sub-agents or steps
within the loop. Callbacks, like after_agent_callback, can also interact with the
State to extract final results or modify the flow based on accumulated
information.15

●  Python Examples (Conceptual Access):

Access to the session state is often provided through a tool_context object when
a tool is called, or directly within the agent's logic if it's designed to manage state.
Python
# Conceptual: Accessing state within a tool function
# def my_custom_tool(user_input: str, tool_context: ToolContext) -> str:
#     # Read from state
#     previous_value = tool_context.state.get("some_key")
#     # Process user_input and previous_value
#     new_value = f"{previous_value}_{user_input}"
#     # Write to state
#     tool_context.state.set("some_key", new_value)
#     return f"State updated with {new_value}"

# Conceptual: State management in a LoopAgent [15]
# class MyLoopCheckCondition(CheckCondition):
#    def check(self, context: ReadonlyContext) -> bool:
#        # Check a value in the state to decide if loop should continue
#        loop_counter = context.state.get("loop_counter", 0)
#        return loop_counter < 3

# async def my_after_loop_callback(context: ReadonlyContext, state: State) -> None:
#     # Retrieve final result from state after loop finishes
#     final_summary = state.get("final_summary")
#     #...

15

Long-Term Knowledge: The ADK MemoryService

While State handles short-term session context, the ADK MemoryService provides the
interface and implementations for managing a searchable, long-term knowledge
store.28 This is designed for information that needs to persist beyond a single session
or even be shared across multiple users or agents, effectively acting as the agent's
long-term memory.

●  BaseMemoryService Interface:

●

This abstract class defines the contract for all memory service implementations,
ensuring a consistent way for agents to add information to and retrieve
information from long-term storage.28
InMemoryMemoryService:
○  Functionality: This implementation stores session information directly in the
application's memory. Search is typically performed using basic keyword
matching.28

○  Persistence: It offers no persistence; all stored knowledge is lost if the

application restarts.28

○  Use Cases: Best suited for prototyping, simple testing scenarios, or

applications where only rudimentary keyword-based recall is needed and
data persistence is not a requirement.28

○  Python Example:

Python
from google.adk.memory import InMemoryMemoryService
memory_service = InMemoryMemoryService()
28

●  VertexAiRagMemoryService:

This more advanced implementation directly leverages Google's powerful Vertex
AI RAG capabilities, transforming long-term memory from simple keyword
matching into a semantically rich, context-aware knowledge retrieval system. This
is crucial for building truly intelligent agents that can learn and recall relevant
information effectively.
○  Functionality: It ingests session data (or other documents) into a specified

Vertex AI RAG Corpus and utilizes the powerful semantic search capabilities
of RAG for retrieval.28 This means it can understand the meaning and context
behind queries, rather than just matching keywords.

○  Persistence: Knowledge is stored persistently within the configured Vertex AI

RAG Corpus on Google Cloud.28

○  Use Cases: Ideal for production applications that require scalable, persistent,
and semantically relevant knowledge retrieval, especially when deployed on
Google Cloud.28

○  Requirements: A Google Cloud project, appropriate IAM permissions, the
necessary SDKs (installed via pip install google-adk[vertexai]), and a
pre-configured Vertex AI RAG Corpus resource name/ID.28

○  Python Example:

Python
from google.adk.memory import VertexAiRagMemoryService

# The RAG Corpus resource name or ID from your Google Cloud project
RAG_CORPUS_RESOURCE_NAME =
"projects/your-gcp-project-id/locations/us-central1/ragCorpora/your-corpus-id"

# Optional configuration for the embedding model used by the RAG service
# from google.adk.memory import RagEmbeddingModelConfig, VertexPredictionEndpoint
# embedding_config = RagEmbeddingModelConfig(
#     vertex_prediction_endpoint=VertexPredictionEndpoint(
# publisher_model="publishers/google/models/text-embedding-005" # Example
#     )
# )

memory_service = VertexAiRagMemoryService(
    rag_corpus_resource_name=RAG_CORPUS_RESOURCE_NAME
    # Optional: rag_embedding_model_config=embedding_config
)
28

●

Implementing Memory Retrieval Tools in Agents:
Agents typically interact with the MemoryService through dedicated tools. The
ADK documentation provides an example of a built-in load_memory tool.28
1.  Agent Invokes Tool: An agent, equipped with a memory-retrieval tool like

load_memory, recognizes the need for past context based on the user's query
or its internal state. It then calls this tool, providing a search query (e.g., "What
did we discuss about Project X last week?").

2.  Tool Calls MemoryService: The load_memory tool internally calls the

search_memory(...) method of the configured MemoryService instance (e.g.,

InMemoryMemoryService or VertexAiRagMemoryService).

3.  Context Used by Agent: The retrieved information is then provided back to

the agent, which uses it to formulate a more informed response.

Python Example (Using load_memory tool with InMemoryMemoryService):The
following conceptual example is based on the structure provided in ADK
documentation.28Python
from google.adk.agents import Agent
from google.adk.memory import InMemoryMemoryService
from google.adk.tools.memory import load_memory # Built-in tool
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types # For constructing Content objects

# --- Services ---
session_service = InMemorySessionService()
memory_service = InMemoryMemoryService() # Using in-memory for this demo

# --- Agent that captures information and stores it in memory ---
info_capture_agent = Agent(
    name="InfoCaptureAgent",
    model="gemini-2.0-flash",
    description="Captures and remembers user's favorite project.",
    instruction=(
        "Ask the user for their favorite project. "
        "Then, confirm you've remembered it using the save_memory tool."
        # Note: 'save_memory' tool is conceptual here based on 'load_memory' logic.
        # Actual ADK might have a different mechanism or expect the LLM to output
        # structured data that an external process then saves to memory.
        # For simplicity, we'll assume the agent's response itself is added to memory.
    ),
    # tools=[save_memory] # Assuming a save_memory tool exists or is part of LLM output handling
)

# --- Agent that recalls information from memory ---
memory_recall_agent = Agent(
    name="MemoryRecallAgent",
    model="gemini-2.0-flash",
    description="Recalls information from past conversations.",
    instruction=(
        "Answer the user's question based on past conversations. "
        "Use the 'load_memory' tool if the answer might be in past conversations."
    ),
    tools=[load_memory] # Give the agent the load_memory tool
)

# --- Runner Setup ---
# For simplicity, we might use a workflow agent or switch agents.
# Here, we'll simulate two separate interactions.
runner_capture = Runner(
    agent=info_capture_agent,
    app_name="MemoryDemoApp",
    session_service=session_service,
    memory_service=memory_service # Provide memory service to runner
)
runner_recall = Runner(
    agent=memory_recall_agent,
    app_name="MemoryDemoApp",
    session_service=session_service,
    memory_service=memory_service # Provide memory service to runner
)

USER_ID = "test_user"
SESSION_ID_CAPTURE = "session_capture_01"
SESSION_ID_RECALL = "session_recall_01" # Could be same or different session

# --- Interaction 1: Capture Information ---
# print("Running InfoCaptureAgent...")
# user_input_capture = types.Content(parts=[types.Part(text="My favorite project is 'Phoenix'.")],
role="user")
# for event in runner_capture.run(user_id=USER_ID, session_id=SESSION_ID_CAPTURE,
new_message=user_input_capture):
#     if event.is_final_response():
#         print(f"Capture Agent: {event.content.parts.text}")
#         # Manually add conversation turn to memory for demo purposes
#         # In a real scenario, this might be automated or handled by specific tools/callbacks.
#         # The 'load_memory' tool typically searches conversation history stored by the MemoryService.
#         # The ADK framework might automatically log turns to the MemoryService if configured.
#         # The provided [28] example implies MemoryService ingests session data.
#         # Let's assume the runner or a callback handles adding conversation turns to memory_service.
#         # For VertexAiRagMemoryService, this involves ingesting session data.

# --- Interaction 2: Recall Information ---
# print("\nRunning MemoryRecallAgent...")
# user_input_recall = types.Content(parts=[types.Part(text="What is my favorite project?")],
role="user")
# for event in runner_recall.run(user_id=USER_ID, session_id=SESSION_ID_RECALL,
new_message=user_input_recall):
#     if event.is_final_response():
#         print(f"Recall Agent: {event.content.parts.text}")
28The example from 28 focuses more on the load_memory tool and assumes that
conversation history or relevant data has been populated into the MemoryService.
The VertexAiRagMemoryService specifically mentions ingesting session data into a
RAG Corpus.

Vertex AI Agent Builder's Memory (General Context)

It's important to note that while this section details ADK's specific memory
components (State, MemoryService), the broader Vertex AI Agent Builder platform, of
which ADK is a part, also emphasizes comprehensive memory capabilities. For
instance, the Vertex AI Agent Engine is explicitly stated to support both short-term
and long-term memory, ensuring that agents deployed on it can maintain context and
recall past interactions.10 This platform-level commitment to memory ensures that
ADK-developed agents can leverage robust memory solutions when deployed.

Table 4.1: ADK Memory Solutions Comparison

Memory
Type

ADK
Component
/Class

Persistence

Scalability

Key
Characteris
tics

Ideal Use
Case

Short-Term
Session
State

State object
(within a
Session)

Session-bou
nd

Per session
(lost when
session
ends)

Long-Term
In-Memory

InMemoryMe
moryService

Limited by
app memory

None (lost
on
application
restart)

Mutable, for
temporary
data within a
single
conversation
or workflow.
Fast access.

Basic
keyword
search, no
persistence.
Simple to set
up.

Maintaining
context
across turns
in a
dialogue,
passing data
between
steps in a
short-lived
workflow.

Prototyping,
testing,
simple
scenarios
where
persistence
and
advanced
search are
not critical.
28

Long-Term
Vertex AI

VertexAiRag
MemoryServi

Persistent
(via Vertex AI

Cloud-scala
ble

Semantic
search via
RAG,

Production
applications
needing

RAG

ce

RAG Corpus)

persistent
storage on
GCP,
leverages
powerful
Vertex AI
capabilities.
Requires
GCP setup.

scalable,
persistent,
and
semantically
relevant
long-term
knowledge.
28

This table provides a clear comparison to help developers select the most suitable
memory management strategy based on their agent's specific requirements for
context retention, data persistence, and scalability.10

5. Building Collaborative Intelligence: Agent-to-Agent (A2A)
Communication

Introduction to the A2A Protocol (v0.2) and its Significance

As AI agent systems become more sophisticated, the need for individual agents to
collaborate and communicate effectively becomes paramount. The Agent-to-Agent
(A2A) communication protocol is an open standard designed by Google and its
partners to address this challenge. It aims to enable seamless and standardized
interaction between diverse AI agents, irrespective of the underlying framework (e.g.,
ADK, LangGraph, CrewAI) or vendor they are built on.4 This focus on an open standard
is a strategic move to foster a larger, interconnected agent ecosystem rather than a
collection of siloed, proprietary solutions.

A significant update to the A2A protocol, version 0.2, was rolled out around May
2025.4 This version introduced key enhancements:

●  Support for Stateless Interactions: This simplifies development for scenarios
where ongoing session management between agents is not required, leading to
more efficient and lightweight communication.4

●  Standardized Authentication Schemes: Version 0.2 incorporates an
OpenAPI-like authentication schema, ensuring clear communication of
authentication requirements (e.g., API keys, OAuth tokens) across agents. This
bolsters security and reliability in inter-agent interactions.4

The A2A protocol is positioned to complement the Model Context Protocol (MCP).
While MCP focuses on standardizing how agents connect to tools and data sources,
A2A is centered on enabling agents to collaborate with each other using their natural

modalities (e.g., conversational exchange) rather than treating each other merely as
tools.30 The growing industry adoption of A2A, with over 50 partners including major
players like Auth0, Box, Microsoft, SAP, and Zoom, underscores its potential to
become a cornerstone for building complex, interoperable multi-agent systems.4

Core A2A Concepts: Agent Cards, Task Management, Authentication

The A2A protocol revolves around several core concepts that define how agents
discover, interact, and manage tasks:

●  Agent Card:

An AgentCard serves as a public, machine-readable description of an
A2A-enabled agent's capabilities.15 It is typically exposed via a well-known URI
(e.g., /.well-known/agent.json) on the A2A server. The Agent Card includes:
○  Basic information: name, description, url (endpoint of the agent), version.
○  Communication modes: defaultInputModes and defaultOutputModes (e.g.,

"text/plain", "application/json").

○  Capabilities: Flags indicating support for features like streaming or

pushNotifications.

○  Skills: A list of AgentSkill objects, each detailing a specific capability of the

agent with id, name, description, tags, and examples. This helps other agents
understand what tasks this agent can perform.

○  Authentication: authenticationSchemes supported by the agent (e.g., "Basic",
"Bearer" for API keys). This structured information allows an orchestrator or
client agent to make informed decisions about which remote agent to
delegate a task to, based on advertised skills and compatible communication
methods, moving beyond simple hardcoded API calls.

●  Task Management API:

A2A communication for task delegation typically follows a JSON-RPC based
standard.30 An A2A client sends a task to an A2A server via a method like
tasks/send. The server then manages the lifecycle of this task, which can
transition through states such as WORKING, COMPLETED, INPUT_REQUIRED (if
the agent needs more information from the user/client), CANCELED, or FAILED.30
The server can also send push notifications to the client about task progress if
configured.
●  Authentication:

As mentioned, A2A v0.2 standardizes authentication. The Agent Card declares the
supported schemes (e.g., HTTP Basic Authentication, Bearer Token for API Key
authentication), and the client must adhere to these when sending requests.4

●  Communication Patterns:

A2A supports various communication patterns, including simple
request-response, streaming of responses for real-time updates (as
demonstrated in a LangGraph A2A example 32), and multi-turn dialogues, often
facilitated by the INPUT_REQUIRED task state.15 The introduction of stateless
interactions in v0.2 further simplifies scenarios where a persistent session
between agents is unnecessary.4

Developing A2A-Enabled Agents with the Python SDK

To simplify the implementation of A2A communication in Python-based agents,
Google released an official Python SDK for A2A around May 2025.4 This SDK provides
tools and abstractions for both exposing an ADK agent as an A2A server and for an
ADK agent to act as an A2A client. The InstaVibe Codelab 15 and the Purchasing
Concierge Codelab 30 offer invaluable practical Python implementations.

Exposing an ADK Agent as an A2A Server (Python example)

To make an ADK agent accessible to other agents via A2A, it needs to be wrapped
with an A2A server component.

1.  AgentWithTaskManager Wrapper:

A common pattern is to create a wrapper class (e.g., PlannerAgent in the
InstaVibe Codelab 15) that inherits from AgentWithTaskManager (or a similar base
from the A2A SDK/ADK). This class bridges the A2A server's task handling logic
with the underlying ADK agent's execution.
Python
# Conceptual structure based on [15] (InstaVibe PlannerAgent)
# from google.adk.a2a import AgentWithTaskManager # Assuming such a base class or similar
# from google.adk.agents import Agent, Runner
# from google.adk.sessions import InMemorySessionService
# from. import agent as planner_adk_agent # Your core ADK agent logic

# class MyA2AAgentWrapper(AgentWithTaskManager):
#     SUPPORTED_CONTENT_TYPES = ["text/plain"]

#     def __init__(self):
#         self._adk_agent_instance = planner_adk_agent.root_agent # Your ADK agent
#         self._runner = Runner(
#             app_name=self._adk_agent_instance.name,
#             agent=self._adk_agent_instance,
#             session_service=InMemorySessionService(),
#             #... other services like memory, artifact...
#         )
#     async def process_task(self, task_input, session_id, user_id): # Method called by A2A server
#         #... logic to convert task_input to ADK Content

#         #... use self._runner.run_async(...) with the ADK agent
#         #... convert ADK agent's response back to A2A task output format
#         pass
    # Other methods like get_processing_message might be defined

2.  Define AgentCard:

Construct an AgentCard object detailing the agent's name, description, URL (its
public endpoint), version, supported input/output modes, capabilities, skills, and
authentication schemes.15
3.  Instantiate and Start A2AServer:

Use the A2A Python SDK's A2AServer class (or equivalent), providing it with the
AgentCard and an instance of your AgentTaskManager (which uses your wrapped
ADK agent).15
Python
# Conceptual server startup based on [15] and [30]
# from google.adk.a2a.server import A2AServer # Hypothetical SDK import
# from google.adk.a2a.types import AgentCard, AgentCapabilities, AgentSkill # Hypothetical
types
# from.my_a2a_agent_wrapper import MyA2AAgentWrapper, AgentTaskManagerImpl # Your
implementations

# PUBLIC_URL = "https://your-agent-server-url.cloudrun.app" # Example
# HOST = "0.0.0.0"
# PORT = 8080

# agent_card_data = AgentCard(
#     name="MyAwesomeADKAgent",
#     description="This agent does awesome things via A2A.",
#     url=PUBLIC_URL,
#     version="1.0.0",
#     defaultInputModes=["text/plain"],
#     defaultOutputModes=["text/plain"],
#     capabilities=AgentCapabilities(streaming=False), # Example
#     skills=,
#     # authenticationSchemes=[...] # If authentication is needed
# )

# task_manager_instance = AgentTaskManagerImpl(agent_wrapper=MyA2AAgentWrapper())

# server = A2AServer(
#     agent_card=agent_card_data,
#     task_manager=task_manager_instance,
#     host=HOST,
#     port=PORT

# )
# server.start() # This would start the HTTP server

The AgentTaskManager 30 handles incoming tasks/send requests, validates them,
updates an internal task store, invokes the core agent logic, and manages the
response lifecycle.

An ADK Agent as an A2A Client: Sending and Managing Tasks (Python example)

An ADK agent can also act as an A2A client, discovering and delegating tasks to other
A2A-enabled agents. This is typical for an orchestrator agent.

1.  Discover Remote Agents:

Use an A2ACardResolver (from the A2A SDK) to fetch the AgentCard of remote
agents by their URLs. Store these connections, perhaps in a dictionary mapping
agent names to connection objects (like RemoteAgentConnections in ADK).15

2.  Create ADK Tools for A2A Interaction:

Define ADK function tools within the orchestrator agent, such as
list_remote_agents (to inform the LLM about available agents) and send_task (to
delegate tasks).15

3.  send_task Tool Implementation:

This tool would:
○  Accept the target remote_agent_name and the task_details (e.g., user

request, parameters).

○  Retrieve the client connection for the target agent.
○  Construct the A2A TaskSendParams object, including a unique taskId,

sessionId, and a Message object containing the user's query and relevant
metadata.

○  Call the remote agent's tasks/send endpoint (e.g., via client.send_task() from

the SDK).

○  Handle the response: inspect task.status.state (e.g., COMPLETED,

INPUT_REQUIRED), process task.status.message and task.artifacts, and
potentially update the orchestrator's own state or escalate to the user if more
input is needed.15

Python
# Conceptual send_task tool for an ADK orchestrator agent [15]
# from google.adk.tools import FunctionTool, ToolContext
# from google.adk.a2a.client import A2AClient, TaskSendParams, Message, Part # Hypothetical
SDK
# from google.adk.a2a.types import TaskState

# self.remote_agent_connections = {} # Populated during agent init with A2ACardResolver

# async def send_task_to_remote_agent(
#     remote_agent_name: str,
#     user_request: str,
#     session_id: str, # Orchestrator's session ID
#     tool_context: ToolContext
# ) -> str:
#     if remote_agent_name not in self.remote_agent_connections:
#         return f"Error: Remote agent '{remote_agent_name}' not found."

#     a2a_client: A2AClient = self.remote_agent_connections[remote_agent_name].client
#     task_id = f"task_{uuid.uuid4()}" # Generate unique task ID

#     params = TaskSendParams(
#         id=task_id,
#         sessionId=session_id, # Can be orchestrator's session or a new one
#         message=Message(role="user", parts=[Part(type="text", text=user_request)])
#     )

#     try:
#         task_response = await a2a_client.send_task(params=params) # Make the A2A call

#         if task_response.status.state == TaskState.COMPLETED:
#             # Extract and return the result from task_response.status.message or
task_response.artifacts
#             return f"Task completed by {remote_agent_name}:
{task_response.status.message.parts.text}"
#         elif task_response.status.state == TaskState.INPUT_REQUIRED:
#             tool_context.actions.escalate = True # Signal ADK to ask user for more input
#             return f"{remote_agent_name} requires more input:
{task_response.status.message.parts.text}"
#         else:
#             return f"Task status from {remote_agent_name}: {task_response.status.state}"
#     except Exception as e:
#         return f"Error sending task to {remote_agent_name}: {str(e)}"

# # This function would be wrapped as an ADK FunctionTool and added to the orchestrator.

4.  Orchestrator Agent instruction:

The orchestrator agent's system prompt (instruction) would guide it to use these
A2A tools, understand when to discover agents, how to plan sequential tasks, and
how to delegate them based on the capabilities advertised in the remote agents'
cards.15

A2A Communication Patterns and Best Practices

●  Sequential Task Delegation: For multi-step processes, an orchestrator can

delegate tasks to specialized agents sequentially, using the output of one agent
as input for the next.

●  Handling INPUT_REQUIRED: When a remote agent returns a status of

INPUT_REQUIRED, the client (orchestrator) needs to relay this request for more
information back to the end-user and then send the updated information back to
the remote agent. This is key for multi-turn conversational tasks handled by a
remote agent.15

●  Streaming: For long-running tasks or when partial results are beneficial, A2A

supports streaming responses, allowing the client to receive updates
incrementally.32

●  Clear Agent Cards: Well-defined AgentCards with accurate descriptions of skills,
input/output modes, and examples are crucial for effective agent discovery and
selection by orchestrators.

●  Robust Error Handling: Implement comprehensive error handling for network

issues, invalid task states, and timeouts.

●  Task State Management: Both clients and servers need to manage task states

appropriately to ensure reliable interaction.

The detailed Codelabs (InstaVibe and Purchasing Concierge) serve as invaluable
blueprints, providing concrete Python implementations that demonstrate how the A2A
protocol translates into working code within the ADK framework, significantly lowering
the barrier to entry for developers aiming to build collaborative multi-agent systems.

Table 5.1: A2A Protocol v0.2 Highlights (May 2025)

Feature

Description

Developer Impact

Stateless Interactions

Standardized
Authentication

Python SDK

Allows for A2A communication
without requiring persistent
session management between
the interacting agents.

Simplifies development for
many use cases, leading to
more efficient and lightweight
inter-agent communication. 4

OpenAPI-like authentication
schema for declaring and
enforcing auth requirements
(e.g., API keys, OAuth).

Enhances security and
reliability by providing a clear,
standard way for agents to
authenticate each other. 4

Official Software Development
Kit for Python to facilitate A2A
integration.

Lowers the barrier to entry for
Python developers, providing
tools and abstractions for A2A
clients and servers. 4

Agent Card for Discovery

Machine-readable public
description of an agent's
capabilities, skills, URL, and
supported interaction modes.

Enables dynamic agent
discovery and selection by
orchestrators, fostering more
flexible multi-agent systems.
30

JSON-RPC Task
Management API

Standardized API (e.g.,
tasks/send) for clients to
delegate tasks and for servers
to manage task lifecycles.

Provides a consistent protocol
for task delegation, status
tracking, and response
handling between agents. 30

Growing Partner Ecosystem

Adoption by 50+ industry
partners (e.g., Microsoft, SAP,
Box).

Indicates strong momentum
and the potential for
widespread interoperability
across different agent
platforms. 4

This table summarizes the key advancements in A2A v0.2, helping developers
understand its matured capabilities for building robust and interoperable multi-agent
applications.

6. From Development to Production: Vertex AI Agent Engine

Vertex AI Agent Engine: Your Managed Runtime for Scalable Agents

Vertex AI Agent Engine is a fully managed Google Cloud service specifically designed
to deploy, manage, and scale AI agents in production environments.10 Previously
known by names such as LangChain on Vertex AI or Vertex AI Reasoning Engine 34, its
core purpose is to abstract away the complexities of infrastructure management,
allowing developers to concentrate on the unique logic and capabilities of their
agents.34 This unified deployment target simplifies operations for teams, even if they
are using different Python-based agent frameworks.

Agent Engine offers several key features for production deployments:

●  Fully Managed Environment: It provides a robust runtime with built-in security

features, including VPC Service Controls (VPC-SC) compliance, and offers
comprehensive end-to-end management capabilities.34

●  Scalability: The service handles the infrastructure necessary to scale agents

according to demand.12

●  Framework Agnostic: While optimized for ADK and Google's ecosystem, Agent

Engine supports deploying agents built with various Python frameworks, including
LangGraph, LangChain, AG2, LlamaIndex, and even custom templates.10

●  Observability: It integrates with Google Cloud Trace (supporting OpenTelemetry)
for performance monitoring and tracing of agent interactions, which is crucial for
debugging complex or multi-agent applications.29

●  Simplified Development: Agent Engine abstracts low-level tasks like application

server development and authentication/IAM configuration.34

Deploying Python ADK Agents to Agent Engine (Step-by-step with SDK
examples)

The Vertex AI SDK for Python provides a streamlined path for deploying ADK-built
agents to Agent Engine.

1.  Prerequisites 12:

○

○

Install the Vertex AI SDK with necessary extras:
Bash
pip install google-cloud-aiplatform[adk,agent_engines]
Agent Engine supports Python versions 3.9 to 3.12 (inclusive).
Initialize the Vertex AI SDK in your Python script:
Python
import vertexai

PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"  # Check Agent Engine documentation for supported regions
STAGING_BUCKET = "gs://your-gcs-staging-bucket" # For deployment artifacts

vertexai.init(
    project=PROJECT_ID,
    location=LOCATION,
    staging_bucket=STAGING_BUCKET,
)

2.  Prepare Your ADK Agent for Agent Engine 12:

Wrap your root ADK agent (e.g., root_agent defined in Section 2) using
reasoning_engines.AdkApp(). This prepares it for deployment and allows enabling
tracing.
Python
from vertexai.preview import reasoning_engines
# Assuming 'root_agent' is your defined ADK Agent instance
# from multi_tool_agent.agent import root_agent # Example import

app_for_deployment = reasoning_engines.AdkApp(

    agent=root_agent,
    enable_tracing=True, # Recommended for observability
)

3.  Deploy the Agent 12:

Use agent_engines.create() to deploy the prepared application to Agent Engine.
This step can take several minutes as it provisions resources and deploys the
agent container.
Python
from vertexai import agent_engines

# Define requirements for your agent's environment
# At a minimum, include the ADK and Agent Engine packages
agent_requirements = [
    "google-cloud-aiplatform[adk,agent_engines]",
    # Add any other specific dependencies your agent tools might need
    # e.g., "tavily-python" if using Tavily search tool
]

remote_deployed_app = agent_engines.create(
    agent_engine=app_for_deployment, # The AdkApp instance
    requirements=agent_requirements  # List of pip requirements
    # display_name="MyDeployedADKAgent" # Optional display name
)
print(f"Agent deployed. Resource name: {remote_deployed_app.resource_name}")

Each deployed agent receives a unique resource name.

4.  Interact with the Deployed Agent 12:

Once deployed, you can interact with the remote agent similarly to how you test
locally, using the remote_deployed_app object.
Python
# Create a session with the deployed agent
# remote_session = remote_deployed_app.create_session(user_id="prod_user_001")
# print(f"Created remote session: {remote_session['id']}")

# Send a query to the deployed agent (streaming example)
# query_message = "What is the weather in New York?"
# for event in remote_deployed_app.stream_query(
#     user_id="prod_user_001",
#     session_id=remote_session["id"],
#     message=query_message

# ):
#     if event.is_final_response():
#         print(f"Deployed Agent Response: {event.content.parts.text}")
#     elif event.get_function_calls():
#         print(f"Deployed Agent called tool: {event.get_function_calls().name}")

5.  Clean Up 12:

To avoid ongoing charges, delete the deployed Agent Engine instance when no
longer needed.
Python
# remote_deployed_app.delete(force=True) # force=True deletes child resources like sessions
# print("Deployed agent deleted.")

A similar deployment process can be used for agents built with other frameworks
like LangChain, as shown by examples deploying Llama 3.1 agents.36

The Agent Engine UI: Managing and Monitoring Deployed Agents

Announced in May 2025, the Agent Engine UI provides a significant enhancement for
managing and observing agents deployed on Vertex AI.4 Accessible directly within the
Google Cloud console, this user-friendly interface aims to simplify the entire agent
lifecycle.

Key functionalities of the Agent Engine UI include 4:

●  Comprehensive Dashboard: A centralized view of all deployed agents.
●  Agent Management: Capabilities to view and manage the deployed agents.
●  Session Listing: Ability to list active and past interaction sessions.
●  Tracing and Debugging: Tools to trace agent actions and debug issues, crucial

for understanding complex agent behavior.

●  Monitoring: Dashboards and metrics to monitor agent performance and health.

This UI provides a more intuitive and centralized way to control and gain deeper
insights into agent behavior and performance, which is essential for production
readiness.

Architectural Considerations for Production Agents

Deploying agents to production requires careful consideration of several architectural
aspects:

●  Scalability: Agent Engine is designed to handle the scaling of agent

infrastructure.12 Design agents to be stateless where possible, relying on session
state and long-term memory services for context.

●  Observability and Tracing: As agents can involve multiple LLM calls, tool

executions, or even A2A interactions, robust tracing is critical. Agent Engine's
integration with Google Cloud Trace (OpenTelemetry) allows developers to
visualize the entire execution flow, identify bottlenecks, and debug errors
effectively.29 The InstaVibe Codelab also highlights using Cloud Trace for
performance analysis of a multi-agent system.15

●  Security: Leverage Agent Engine's support for VPC Service Controls to enhance
data security and mitigate risks of data exfiltration when agents handle sensitive
information.34 Ensure proper IAM permissions are configured for agent service
accounts and access to tools or data sources.

●  Error Handling and Resilience: Agents and their tools must be designed with

robust error handling. Tools should gracefully manage failures (e.g., API timeouts,
invalid inputs) and provide clear error messages back to the agent. The agent, in
turn, should be instructed on how to handle tool failures—whether to retry, ask
the user for clarification, or try an alternative approach.

●  Version Control and CI/CD: Treat agent code (ADK definitions, tool

implementations, instructions) like any other software artifact. Use version control
systems (e.g., Git) and implement CI/CD pipelines for automated testing and
deployment to Agent Engine. General agent building advice also emphasizes
defining clear goals, limitations, and success metrics before development.35

Ensuring Reliability: Monitoring Agents in Production

Once deployed, continuous monitoring is essential to ensure the reliability,
performance, and quality of AI agents. Vertex AI Agent Engine integrates seamlessly
with Cloud Monitoring, providing a rich set of tools for this purpose without requiring
additional setup.38

Key Built-in Metrics

Agent Engine automatically collects several built-in metrics associated with the
aiplatform.googleapis.com/ReasoningEngine monitored resource. These include 38:

●  reasoning_engine/request_count: The number of requests made to the agent,

often labeled by response code.

●  reasoning_engine/request_latencies: Distribution of request processing times.
●  reasoning_engine/container_cpu_allocation_time: CPU time allocated to the

agent's container.

●  reasoning_engine/container_memory_allocation_time: Memory allocated.

Leveraging Cloud Monitoring

●  Metrics Explorer: This Google Cloud console tool allows for visualizing these
metrics, applying filters (e.g., by reasoning_engine_id, response_code), and
aggregating data.38

●  Querying with MQL/PromQL: For more advanced analysis, Monitoring Query
Language (MQL) and Prometheus Query Language (PromQL) can be used to
create custom queries, define custom time intervals, and calculate derived
metrics like error rates.38
○  MQL Example (Error Rate for a specific agent):

Code snippet
fetch aiplatform.googleapis.com/ReasoningEngine

| metric 'aiplatform.googleapis.com/reasoning_engine/request_count'
| filter resource.reasoning_engine_id == 'your-deployed-agent-resource-id'
| {
filter metric.response_code == '500' ; // Count 500 errors
ident
}
| align rate(10m)
| every 10m
| group_by,
[value_request_count_aggregate: aggregate(value.request_count)]
| ratio
```
38
●  Cloud Monitoring API: Programmatic access to metrics is available via the Cloud
Monitoring v3 API, allowing for integration with custom dashboards or automated
reporting systems.38

Implementing Custom Metrics and Alerting Strategies

●  Custom Metrics:

○  Log-based Metrics: Create metrics based on patterns in agent logs. For

example, if a tool writes a specific log entry when called, a log-based metric
can count these occurrences.38

○  User-defined Metrics: For application-specific data not captured by logs
(e.g., number of tokens processed, business-specific KPIs), define custom
metric types and write data points to them using the Cloud Monitoring API.38

●  Alerting:

Cloud Monitoring allows creating alert policies based on metric thresholds. For
instance, an alert can be configured to notify administrators if the 99th percentile

of request_latencies for an agent exceeds a defined threshold (e.g., 5000ms).38

●  Vertex AI Agent Engine Overview Dashboard:

A default dashboard named "Vertex AI Agent Engine Overview" is available in
Cloud Monitoring, providing a starting point for visualizing key operational
metrics. This dashboard can be copied and customized to include custom metrics
and alerts.38

This comprehensive suite of monitoring and management tools is critical for
enterprises to confidently deploy, operate, and maintain AI agents at scale, ensuring
their reliability and enabling continuous improvement based on observed
performance.

Table 6.1: Key Agent Engine Monitoring Metrics

Metric Name
(Partial)

Description

Monitored
Resource

Access Method
Examples

Common
Labels

reasoning_engin
e/request_count

reasoning_engin
e/request_latenc
ies

Number of
requests
processed by
the agent.

Distribution of
time taken to
process
requests.

reasoning_engin
e/container_cpu
_allocation_time

CPU time
allocated to the
agent's
container.

reasoning_engin
e/container_me
mory_allocation
_time

Memory
allocated to the
agent's
container.

logging/user/<cu
stom_log_metric
>

Custom metric
derived from
agent log
entries (e.g.,
tool call counts).

aiplatform.googl
eapis.com/Reas
oningEngine

Metrics Explorer,
MQL, PromQL,
API

response_code,
reasoning_engin
e_id, location

aiplatform.googl
eapis.com/Reas
oningEngine

Metrics Explorer,
MQL, PromQL,
API

reasoning_engin
e_id, location

aiplatform.googl
eapis.com/Reas
oningEngine

Metrics Explorer,
MQL, PromQL,
API

reasoning_engin
e_id, location

aiplatform.googl
eapis.com/Reas
oningEngine

Metrics Explorer,
MQL, PromQL,
API

reasoning_engin
e_id, location

aiplatform.googl
eapis.com/Reas
oningEngine

Metrics Explorer,
MQL, PromQL,
API

User-defined
(e.g., tool_id,
agent_id)

custom.googlea
pis.com/<custo
m_metric>

User-defined
metric for
application-spe
cific data (e.g.,
token counts).

generic_node
(typically)

Metrics Explorer,
MQL, PromQL,
API

User-defined
(e.g.,
model_name,
agent_name)

This table provides developers with a quick reference to essential metrics, aiding in
the diagnosis of issues, performance optimization, and ensuring the reliability of
production AI agents.38

7. The Unified Platform: Vertex AI Agent Builder and AgentSpace

Google's strategy for AI agents extends beyond individual development kits and
deployment engines to encompass a unified platform experience. This is primarily
manifested through Vertex AI Agent Builder, which serves as the comprehensive suite
for developers, and Google Agentspace, which acts as the enterprise-facing hub for
employees to discover and utilize these AI agents.

Vertex AI Agent Builder: An Integrated Suite for Agent Development

Vertex AI Agent Builder is the overarching umbrella for a suite of features designed to
facilitate the discovery, building, and deployment of AI agents within the Vertex AI
platform.13 It aims to provide an end-to-end solution for agent development.

The core components of Vertex AI Agent Builder include 13:

●  Agent Garden: A repository of pre-built agent samples, tools, and templates to

accelerate development.

●  Agent Development Kit (ADK): The open-source, code-first framework
(primarily Python) for building sophisticated agents with precise control.

●  Agent Tools: A rich ecosystem of built-in tools (Google Search, Vertex AI Search,

Code Execution), RAG Engine, Google Cloud tool integrations (Application
Integration, Connectors), MCP tools, and support for third-party tools
(LangChain, CrewAI).

●  Vertex AI Agent Engine: The fully managed runtime for deploying, managing,

and scaling these agents in production.

The typical workflow envisioned within Agent Builder is 13:

1.  Discover: Explore Agent Garden for relevant samples and tools.
2.  Build & Test: Use the ADK to develop and iterate on agent logic and tool

integrations.

3.  Deploy: Deploy the finalized agent to Vertex AI Agent Engine for production use.

An architecture diagram provided in the Vertex AI Agent Builder overview illustrates
how these components interrelate within what was formerly termed "AI Applications".13
This integrated suite approach signifies Google's intent to provide developers with a
cohesive environment, streamlining the journey from agent conception to production
deployment.

AgentSpace: Empowering the Enterprise with AI Agents

Google Agentspace is designed as an enterprise search and AI agent hub, aimed at
transforming how employees work by connecting their everyday applications to
Google-quality multimodal search and the power of AI agents.5 It allows users to
quickly find information across disparate business systems, synthesize insights from
various sources, and take action through pre-built or custom agents, all within an
enterprise-grade security and privacy framework.

Key features of Agentspace include:

●  Application Connectivity: Easily connects to popular enterprise applications like
Confluence, Google Drive, Jira, Microsoft SharePoint, ServiceNow, and more,
enabling information retrieval and action across these systems.8

●  Multimodal Enterprise Search: Leverages Google's advanced search

technology to find and make actionable information distributed across various
enterprise data sources, respecting existing access controls and permissions.39

●  Gemini Integration: Uses Google's Gemini models to summarize, generate

content, and act on enterprise data and web information securely.8

Google Agentspace Enterprise is a specific offering with enhanced features 40:

●

Identity Mapping (GA May 2025): Allows mapping of enterprise identity
providers (IDPs) to external identities from third-party SaaS applications, ensuring
correct access control enforcement.

●  App Tiers by License Type: App capabilities (search vs. search + assistant

features) are determined by the user's Agentspace Enterprise or Enterprise Plus
license.

●  NotebookLM Enterprise Integration: Users can search their NotebookLM

notebooks as a data source within Agentspace.

The Agent Gallery (in AgentSpace): Discovering, Using, and Publishing Agents

A central component of Agentspace is the Agent Gallery. This serves as a hub where
enterprise employees can discover, access, and manage AI agents provided by

Google, developed by internal teams, or offered by third-party partners.5 The Agent
Gallery became generally available (GA) in Google Agentspace Enterprise as of April
2025.40

A significant aspect of the Agent Gallery is its integration with the Google Cloud
Marketplace.6 This allows enterprises to easily find and deploy a wider array of
specialized agents developed by Google's partners, further enriching the capabilities
available through Agentspace. The Futurum Group noted that as of Next 2025, there
were over 130 AI Agent listings in the broader Marketplace.6

Developer Workflow: Publishing ADK Agents to the Agent Gallery

A key bridge between the developer-focused Vertex AI Agent Builder and the
enterprise-user-focused Agentspace is the ability to publish agents. Agents built
using the ADK and deployed on the Vertex AI Agent Engine can be registered and
made available in Google Agentspace, specifically within its Agent Gallery.10 This
allows sophisticated, custom-coded agents developed by technical teams to be
discoverable and usable by non-technical employees throughout the organization,
democratizing access to powerful AI tools.

Agent Designer (in AgentSpace): Enabling No-Code Agent Creation (Contextual
Overview)

Complementing the code-first ADK, Agentspace also introduces Agent Designer, a
no-code interface that empowers employees, regardless of their technical
background, to create custom AI agents tailored to their specific needs and
workflows.5 Users can define agent tasks using natural language descriptions and
connect them to enterprise data sources. As of April/May 2025, Agent Designer was in
preview or private preview, available via an allowlist.8 Agents built with Vertex AI Agent
Builder (which includes ADK) can also be published to Agentspace, indicating a
two-pronged approach to agent creation: pro-code for developers and no-code for
business users. This dual approach caters to different user personas within an
organization, accelerating AI adoption.

Spotlight on Prebuilt Agents (As of May 21, 2025)

Google is populating the Agent Gallery with powerful "banner" or "expert" pre-built
agents to provide immediate value and showcase the potential of its agent
technology. These prebuilt agents are designed to tackle common complex enterprise
tasks, effectively lowering the barrier to entry for AI adoption. Their availability, often
initially through allowlists or preview programs, suggests a phased rollout strategy to

gather feedback and ensure stability before wider general availability.

●  The Deep Research Agent:

○  Capabilities: This agent can autonomously explore complex topics by

synthesizing information from both internal enterprise sources and external
public data. It aims to deliver comprehensive and easily digestible reports
based on a single natural language prompt.5 As of May 2025, it allows users to
upload personal files (PDFs, images) for inclusion in its research, with plans to
extend this to Google Drive and Gmail content.41

○  Availability: Generally available via allowlist as of April/May 2025.8

●  The Idea Generation Agent:

○  Capabilities: Designed to assist employees in brainstorming and generating

novel ideas across various domains. A unique feature is its competitive
evaluation system, inspired by the scientific method, to identify and rank the
most promising solutions.5

○  Availability: In private preview with allowlist as of April/May 2025.8

●  Other Relevant Agentic Capabilities (Contextual):

While not explicitly "AgentSpace prebuilt agents" in the same category, other
Google initiatives demonstrate advanced agentic capabilities that align with this
broader vision:
○  Project Mariner: A research prototype described as a browser-based agentic

AI capable of handling up to 10 different tasks simultaneously, such as
booking flights, conducting research, and shopping. Access is planned for
Google AI Ultra subscribers in the US first.42

○  Gemini "Agent Mode": An experimental feature announced for the Gemini

app, designed to enable the AI to handle complex tasks and planning
autonomously on the user's behalf. It aims to execute multi-step actions with
minimal oversight. This mode is also initially slated for Google AI Ultra
subscribers, with plans to extend these capabilities to Chrome, Search, and
the Gemini platform.41

○  Customer Engagement Suite Agents: Google has enhanced its Customer
Engagement Suite (formerly Contact Center AI) with capabilities to quickly
create AI agents for self-service using natural language, leveraging Gemini's
multimodal models for richer interactions.6

○  AI Shopping Agent: Demonstrated at Google I/O 2025, this agent can assist
with product discovery based on descriptions, track item availability and price
changes, and even facilitate virtual try-on features.46

Table 7.1: AgentSpace Banner Prebuilt Agents and Key Agentic Initiatives (May

21, 2025)

Agent Name /
Initiative

Core
Capabilities

Primary Use
Cases

Key References

5

5

42

41

Availability
Status (as of
May 2025)

Generally
Available via
Allowlist (in
AgentSpace).

Complex
research,
market analysis,
knowledge
discovery.

Innovation
workshops,
problem-solving
, brainstorming.

Private Preview
via Allowlist (in
AgentSpace).

Complex
personal/work
task automation
across web
applications.

Research
prototype; to be
available to
Google AI Ultra
subscribers in
the US.

Personal
assistant for
complex
planning,
scheduling, and
task execution.

Experimental;
initial access for
Google AI Ultra
subscribers; to
extend to
Chrome, Search,
Gemini.

Deep Research
Agent

Idea
Generation
Agent

Project Mariner

Gemini "Agent
Mode"

Autonomously
explores topics,
synthesizes
internal/external
info into reports,
uses personal
files (PDFs,
images).

Assists in
generating novel
ideas, evaluates
them through a
competitive
system.

Browser-based
agentic AI for
handling
multiple
concurrent
tasks (booking,
research,
shopping).

Experimental
Gemini app
feature for
autonomous
planning and
execution of
multi-step tasks
with minimal
oversight.

Customer
Engagement

Quickly create
AI agents for

Customer
support

Enhancements
to Google

6

Suite Agents

AI Shopping
Agent

customer
self-service
using natural
language,
leveraging
multimodal
Gemini models.

Product
discovery from
descriptions,
virtual try-on,
price tracking,
availability
notifications.

automation, FAQ
handling,
service request
processing.

Customer
Engagement
Suite.

Enhanced
e-commerce
experiences,
personalized
shopping
assistance.

46

New features in
Google Search
AI Mode and
Google Labs.

This table provides a snapshot of the prominent prebuilt and emerging agentic
capabilities within Google's ecosystem, illustrating the company's commitment to
providing both ready-to-use AI solutions and powerful platforms for custom agent
development.

8. Conclusion: The Future of Agentic AI with Google Cloud

Recap: Building Sophisticated AI Agents with Google's Python Ecosystem

This guide has navigated the comprehensive suite of tools and services offered by
Google Cloud for creating, deploying, and managing AI agents, with a particular focus
on Python-based development as of May 2025. The journey begins with the Agent
Development Kit (ADK), an open-source, code-first framework that empowers
developers with fine-grained control over agent behavior and logic. Python ADK
v1.0.0 stands as a production-ready foundation, complemented by an emerging Java
ADK.

Agents built with ADK are then equipped with a rich ecosystem of tools, ranging
from built-in functionalities like Google Search, Vertex AI Search, and Code Execution,
to advanced Retrieval Augmented Generation via the Vertex AI RAG Engine.
Connectivity to enterprise systems is robust, enabled by Application Integration and
Integration Connectors for over 100 applications, alongside support for open
standards like the Model Context Protocol (MCP). Furthermore, ADK's extensibility
allows seamless integration of tools from popular open-source frameworks like
LangChain and CrewAI.

Effective conversational AI hinges on memory. ADK addresses this with short-term

session State management and long-term knowledge retention via its MemoryService,
offering both InMemoryMemoryService for prototyping and the powerful
VertexAiRagMemoryService for persistent, semantically rich recall.

For building collaborative intelligence, the Agent-to-Agent (A2A) communication
protocol (v0.2), supported by a Python SDK, enables diverse agents to discover each
other (via Agent Cards) and work together, irrespective of their underlying
frameworks.

When agents are ready for production, Vertex AI Agent Engine provides a fully
managed, scalable, and observable runtime. Its framework-agnostic nature, coupled
with the new Agent Engine UI and deep integration with Cloud Monitoring, ensures
that agents can be reliably deployed and maintained.

Finally, Vertex AI Agent Builder serves as the unified suite encompassing these
components, while Google Agentspace and its Agent Gallery provide the
enterprise-facing platform for employees to discover, utilize, and even create (via
Agent Designer) AI agents, including powerful pre-built solutions like the Deep
Research and Idea Generation agents. Python remains the central language
throughout this developer-focused ecosystem, enabling rapid innovation and robust
agent creation.

Emerging Trends and the Path Forward

The landscape of AI agents, particularly within the Google Cloud ecosystem, is
dynamic and poised for significant advancements. Several emerging trends point
towards an even more powerful and integrated future:

1.  Increased Autonomy and Proactivity: Initiatives like Project Mariner, designed

to handle multiple concurrent tasks, and Gemini's "Agent Mode," aimed at
autonomous multi-step planning and execution, signal a clear trajectory towards
agents that are more proactive and require less human oversight.41 These
"thinking models" are expected to reason through complex problems and take
initiative on behalf of users.

2.  Strengthening of Open Standards: The emphasis on open standards like A2A

and MCP is critical for fostering a federated and interoperable agent ecosystem.4
As more vendors and frameworks adopt these protocols, the ability to create truly
collaborative multi-agent systems that span organizational and technological
boundaries will increase significantly. The growing number of partners supporting
A2A is a strong indicator of this trend.6

3.  Convergence of No-Code and Pro-Code Development: The co-existence and

complementarity of the developer-focused ADK and the no-code Agent Designer
within Agentspace highlight a strategic approach to empower a broader
spectrum of creators.8 This allows domain experts to quickly build tailored agents
for their needs, while developers can tackle more complex, custom integrations
and functionalities. The ability to publish ADK-built agents to AgentSpace further
bridges this gap.

4.  Ever-More Powerful Foundation Models: The capabilities of AI agents are

intrinsically linked to the power of their underlying foundation models. Google's
continued investment in advancing its Gemini series (e.g., Gemini 2.5 Pro and
Flash with enhanced reasoning, multimodal understanding, and tool use) will
directly translate to more intelligent, capable, and nuanced agents.5 Features like
"Deep Think" in Gemini 2.5 Pro for complex mathematical and coding tasks are
indicative of this progress.43

5.  Deeper Ecosystem Integrations and Sophisticated Orchestration: The path

forward will likely involve even tighter integrations between various Google Cloud
services and agent capabilities. This could manifest as more sophisticated
orchestration tools within ADK or Agent Engine, allowing for more complex
conditional logic, parallel execution, and dynamic agent composition. The focus
will be on enabling agents to not just perform isolated tasks, but to manage and
execute entire end-to-end business processes.

Developers and organizations engaging with Google's AI agent ecosystem can
anticipate a future where agents become increasingly integral to workflows,
decision-making, and innovation. By leveraging the robust Python tools, open
standards, and scalable infrastructure detailed in this guide, they are well-positioned
to build the next generation of intelligent assistants.

Works cited

1.  Comprehensive Guide to Building AI Agents Using Google Agent Development

Kit (ADK), accessed May 21, 2025,
https://www.firecrawl.dev/blog/google-adk-multi-agent-tutorial

2.  The Best AI Agent Builder Platform 2025 - Unleash.so, accessed May 21, 2025,

https://www.unleash.so/post/the-best-ai-agent-builder-platform-2025
3.  Write AI agents in Java — Agent Development Kit getting started guide -

Guillaume Laforge, accessed May 21, 2025,
https://glaforge.dev/posts/2025/05/20/writing-java-ai-agents-with-adk-for-java-g
etting-started/

4.  What's new with Agents: ADK, Agent Engine, and A2A Enhancements, accessed

May 21, 2025,
https://developers.googleblog.com/en/agents-adk-agent-engine-a2a-enhancem

ents-google-io/

5.  5 key AI announcements from Google Cloud Next 2025 - SADA, accessed May 21,

2025,
https://sada.com/blog/5-key-ai-announcements-from-google-cloud-next-2025/

6.  Google Cloud Next 2025: The Yellow Brick Road to AI Transformation - The

Futurum Group, accessed May 21, 2025,
https://futurumgroup.com/insights/google-cloud-next-2025-the-yellow-brick-ro
ad-to-ai-transformation/

7.  Google Cloud Next 2025: News and updates, accessed May 21, 2025,

https://blog.google/products/google-cloud/next-2025/

8.  Google Agentspace Gets Smarter and Easier to Build Agents and ..., accessed

May 21, 2025,
https://aragonresearch.com/google-agentspace-gets-smarter-easier-to-build/

9.  Quickstart - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/get-started/quickstart/

10. Build and manage multi-system agents with Vertex AI | Google Cloud Blog,

accessed May 21, 2025,
https://cloud.google.com/blog/products/ai-machine-learning/build-and-manage-
multi-system-agents-with-vertex-ai

11. Get Started - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/get-started/

12. Agent Engine - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/deploy/agent-engine/

13. Vertex AI Agent Builder overview - Google Cloud, accessed May 21, 2025,

https://cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview

14. google/adk-python: An open-source, code-first Python toolkit for building,

evaluating, and deploying sophisticated AI agents with flexibility and control. -
GitHub, accessed May 21, 2025, https://github.com/google/adk-python

15. Google's Agent Stack in Action: ADK, A2A, MCP on Google Cloud, accessed May

21, 2025,
https://codelabs.developers.google.com/instavibe-adk-multi-agents/instructions
?hl=en

16. Google's Agent Stack in Action: ADK, A2A, MCP on Google Cloud, accessed May

21, 2025,
https://codelabs.developers.google.com/instavibe-adk-multi-agents/instructions

17. Tools - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/tools/

18. Built-in tools - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/tools/built-in-tools/

19. Vertex AI release notes | Generative AI on Vertex AI - Google Cloud, accessed

May 21, 2025,
https://cloud.google.com/vertex-ai/generative-ai/docs/release-notes

20. Vertex AI RAG Engine: A developers tool, accessed May 21, 2025,

https://developers.googleblog.com/en/vertex-ai-rag-engine-a-developers-tool/
21. RAG quickstart for Python | Generative AI on Vertex AI | Google Cloud, accessed

May 21, 2025,
https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart
22. arjunprabhulal/adk-vertex-ai-rag-engine: A RAG agent ... - GitHub, accessed May

21, 2025, https://github.com/arjunprabhulal/adk-vertex-ai-rag-engine
23. Google Cloud tools - Agent Development Kit, accessed May 21, 2025,

https://google.github.io/adk-docs/tools/google-cloud-tools/

24. Third party tools - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/tools/third-party-tools/

25. Develop a LangChain agent | Generative AI on Vertex AI - Google Cloud,

accessed May 21, 2025,
https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/develop/lang
chain

26. Model Context Protocol (MCP) - Agent Development Kit - Google, accessed May

21, 2025, https://google.github.io/adk-docs/mcp/

27. 7 AI Agent Builders in 2025: Comprehensive Guide | Generative AI Collaboration

Platform, accessed May 21, 2025, https://orq.ai/blog/ai-agent-builders
28. Memory - Agent Development Kit - Google, accessed May 21, 2025,

https://google.github.io/adk-docs/sessions/memory/

29. Vertex AI Agent Builder | Google Cloud, accessed May 21, 2025,

https://cloud.google.com/products/agent-builder

30. Getting Started with Agent-to-Agent (A2A) Protocol: A Purchasing ..., accessed

May 21, 2025,
https://codelabs.developers.google.com/intro-a2a-purchasing-concierge
31. How to Build Two Python Agents with Google's A2A Protocol - Step by Step

Tutorial, accessed May 21, 2025,
https://docs.kanaries.net/articles/build-agent-with-a2a

32. A2A/samples/python/agents/langgraph/README.md at main · google/A2A -

GitHub, accessed May 21, 2025,
https://github.com/google/A2A/blob/main/samples/python/agents/langgraph/REA
DME.md

33. Build AI Agents with Vertex AI Agent Engine and Atlas - Atlas - MongoDB Docs,

accessed May 21, 2025,
https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/google
-vertex-ai/agent-engine/

34. Vertex AI Agent Engine overview - Google Cloud, accessed May 21, 2025,

https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview
35. Build and Deploy an Agent with Agent Engine in Vertex AI | Google Cloud Skills

Boost, accessed May 21, 2025,
https://www.cloudskillsboost.google/focuses/104687?parent=catalog

36. vertex-ai-samples/notebooks/community/model_garden/model_garden_agent_e

ngine_llama3_1.ipynb at main - GitHub, accessed May 21, 2025,
https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks
/community/model_garden/model_garden_agent_engine_llama3_1.ipynb

37. Building AI Agents with Vertex AI Agent Builder | Google Codelabs, accessed May

21, 2025,

https://codelabs.developers.google.com/devsite/codelabs/building-ai-agents-ver
texai

38. Monitor an agent | Generative AI on Vertex AI | Google Cloud, accessed May 21,

2025,
https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/manage/mo
nitoring

39. Google Agentspace, accessed May 21, 2025,

https://cloud.google.com/products/agentspace

40. Google Agentspace release notes | Google Cloud, accessed May 21, 2025,

https://cloud.google.com/agentspace/docs/release-notes

41. Google supercharges Gemini with new AI features at I/O 2025: 13 Updates you

should not miss | Mint, accessed May 21, 2025,
https://www.livemint.com/technology/tech-news/google-supercharges-gemini-w
ith-new-ai-features-at-i-o-2025-13-updates-you-should-not-miss-117478495817
57.html

42. Everything Google unveiled at I/O 2025: Gemini, AI Search, smart glasses, more |

ZDNET, accessed May 21, 2025,
https://www.zdnet.com/article/everything-google-unveiled-at-io-2025-gemini-ai
-search-smart-glasses-more/

43. Google goes all-in on AI at I/O 2025, launches 20 new products | YourStory,

accessed May 21, 2025,
https://yourstory.com/2025/05/google-goes-all-in-on-ai-at-i-o-2025-launches-2
0-new-products

44. Google I/O 2025: Unveiling AI Mode and Project Mariner - UBOS.tech, accessed

May 21, 2025,
https://ubos.tech/news/google-i-o-2025-unveiling-ai-mode-and-project-mariner/

45. Google Says Gemini's Agent Mode Will Finally Turn Its AI into a Real Personal

Assistant, accessed May 21, 2025,
https://lifehacker.com/tech/google-io-2025-gemini-agent-mode-turns-ai-into-re
al-personal-assistant

46. Everything you need to know from Google I/O 2025 - Mashable, accessed May

21, 2025,
https://mashable.com/article/google-io-2025-everything-you-need-to-know

