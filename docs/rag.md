# RAG Module Documentation (LangGraph)

## Overview
The RAG module implements a LangGraph-based retrieval-augmented generation pipeline with three sequential nodes: Planner → Fetcher → Thinker. This architecture enables intelligent query planning, semantic retrieval, and thoughtful response generation.

## Graph Flow
```
START → [Planner] → [Fetcher] → [Thinker] → [Context] → END
```

### Node Sequence:
1. **Planner**: Analyzes user query and chat history to generate optimized search queries
2. **Fetcher**: Executes semantic search using Redis vector store with optional reranking
3. **Thinker**: Synthesizes retrieved content into coherent, well-reasoned responses
4. **Context**: Manages conversation history persistence and session state

## Node Details - Algorithmic Explanation

### 1. Planner Node (`planner.py`)
**What it actually does:**

**Step-by-step thinking process:**
1. **Understand the conversation**: Looks at the last few chat messages to understand context
2. **Condense the question**: If you ask "What about his sword skills?" after talking about Adol, it understands "What are Adol Christin's sword skills?"
3. **Break it down**: Takes complex questions and creates simpler search queries
   - Input: "Tell me about Adol's abilities and weapons in Ys VIII"
   - Output queries: ["Adol Christin abilities", "Adol Christin weapons", "Ys VIII combat system"]
4. **Find key terms**: Identifies important names and concepts for better filtering
   - Extracts: ["Adol Christin", "Ys VIII", "abilities", "weapons"]
5. **Fallback mode**: If AI isn't available, uses simple keyword extraction instead

**Real example transformation:**
- User asks: "How does the combat work in the latest game?" 
- After seeing history about Ys series, it creates: ["Ys combat system", "latest Ys game mechanics", "real-time combat Ys"]

### 2. Fetcher Node (`fetcher.py`)
**What it actually does:**

**Search process explained:**
1. **Convert to numbers**: Turns each search query into mathematical vectors (384 numbers)
2. **Find similar content**: Searches Redis for content chunks with similar number patterns
3. **Filter by topics**: Uses the key terms from Planner to focus on relevant sections
4. **Quality check**: Reranks results to put the most relevant answers first
5. **Remove duplicates**: Ensures you don't get the same information multiple times

**Scoring system:**
- 0.0 to 1.0 scale where higher numbers mean better matches
- Typically keeps results above 0.4 similarity
- Reranking can adjust scores based on deeper understanding

**Hybrid Search Approach:**
- **Vector Search**: Primary semantic matching using BGE-small-en-v1.5 embeddings
- **Keyword Filtering**: Secondary filtering based on Planner-extracted entities
- **Reranking**: Optional cross-encoder for precision scoring
- **Why Hybrid**: Combines semantic understanding with keyword precision for niche content

### 3. Thinker Node (`thinker.py`)
**What it actually does:**

**Response building process:**
1. **Read everything**: Looks at all the retrieved information chunks
2. **Check if enough**: Decides if there's sufficient information to answer properly
3. **Weave together**: Combines information from different sources into one coherent answer
4. **Be honest**: If information is missing or conflicting, it acknowledges this
5. **Track sources**: Remembers which content chunks were used for the answer

**Decision making examples:**
- If finds 3 good chunks about Adol's abilities → "Yes, I can answer this"
- If only finds vague or unrelated chunks → "I don't have enough specific information"
- If information conflicts between sources → Presents different perspectives

**Fallback behavior:**
- When AI isn't available, uses simple scoring to decide if content is relevant
- Still provides basic answers but with less sophistication

### 4. Context Node (`context.py`)
**What it actually does:**

**History management process:**
1. **Check response quality**: Only saves successful Q&A pairs (non-empty questions and answers)
2. **Persist to storage**: Saves the conversation turn to Redis session history
3. **Reload updated history**: Refreshes the in-memory history with latest persisted data
4. **Maintain session integrity**: Handles storage failures gracefully without breaking the flow

**Session management features:**
- Automatic TTL (time-to-live) management for session data
- LRU (Least Recently Used) windowing to keep only recent conversation history
- Error resilience - continues operation even if Redis is temporarily unavailable
- Trace logging for debugging history persistence operations

## State Management (`schema.py`)

### Core State Classes:
- **RagState**: Main state container for entire RAG flow
- **PlannerState**: Query planning results (condensed query, planned queries, entities)
- **FetcherState**: Retrieved chunks and metadata
- **ThinkerState**: Final response and sufficiency assessment

### State Flow:
```python
# Initial state
state = RagState(user_query="query", history=[])

# After planner
state.planner_state = PlannerState(planned_queries=["query1", "query2"])

# After fetcher  
state.fetcher_state = FetcherState(chunks=[chunk1, chunk2])

# After thinker
state.thinker_state = ThinkerState(answer="final response")
```

## Usage Examples

### Full RAG Flow:
```python
graph = RagGraph()
state = RagState(user_query="Tell me about Adol Christin's abilities")
result = graph.run(state)
print(result.thinker_state.answer)
```

### Individual Node Execution:
```python
# Run just planner
planner = PlannerNode()
planned_state = planner.run(state)

# Run just fetcher  
fetcher = FetcherNode()
retrieved_state = fetcher.run(planned_state)

# Run just thinker
thinker = ThinkerNode()
final_state = thinker.run(retrieved_state)
```

### CLI Usage:
```bash
# Full flow
python -m core.rag.rag --full-flow --query "Ys series timeline"

# Individual phases
python -m core.rag.rag --phase planner --query "Character abilities"
python -m core.rag.rag --phase fetcher --file planner_output.json
python -m core.rag.rag --phase thinker --file fetcher_output.json
```

## Integration Points

### With Embedding Module:
- Uses same BGE-small-en-v1.5 model for consistent embeddings
- Leverages RedisVectorStore for semantic search
- Maintains chunk metadata for filtering

### With History Module:
- Chat history integration for query condensation
- Session-based conversation tracking
- Context preservation across turns

## Performance Features

- **Latency Tracking**: Each node records execution time in milliseconds
- **Warmup**: Pre-loads models and tests connections on initialization
- **Fallback Systems**: Graceful degradation when external services unavailable
- **Batch Processing**: Efficient handling of multiple queries in fetcher

## Configuration

Key settings in environment configuration:
- `PLANNER_AGENT`: Enable/disable LLM-based planning
- `RERANK_ENABLED`: Toggle cross-encoder reranking
- `RAG_HISTORY_WINDOW`: Number of previous turns to consider
- Model names and API endpoints for each component

The LangGraph architecture provides a robust, extensible foundation for the RAG pipeline, enabling clear separation of concerns and easy maintenance of each processing stage.