from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from google import genai
from google.genai import types
import json
import os
from typing import List, Optional, Dict
import markdown
from enum import Enum
from datetime import datetime, timedelta
import uuid
import time
import conversation_logger


# ============================================================================
# SECTION 1: ROLE DEFINITIONS (NEW - Add after imports)
# ============================================================================

class UserRole(Enum):
    """Define the five role types that determine content filtering"""
    POLICY_MAKER = "policy_maker"          # Policy maker/funding agency
    CONSULTANT = "consultant"               # Consultant/system integrator
    MPO_STAFF = "mpo_staff"                # MPO or transit agency staff
    PLANNER = "planner"                    # Transportation planner
    ENGINEER = "engineer"                  # Traffic operations engineer
    UNKNOWN = "unknown"                    # Default fallback

# Role-based content type configuration
# This controls WHAT content each role can see
# NOTE: Limits increased 2-3x due to smaller chunk sizes (chunks vs full documents)
ROLE_CONTENT_CONFIG = {
    UserRole.POLICY_MAKER: {
        'content_types': ['plandetail', 'spinstance', 'funreq'],  # NO standards, NO interfaces
        'response_style': 'executive',
        'max_results_per_type': {
            'plandetail': 50,   # was 20 (2.5x increase)
            'spinstance': 50,   # was 20 (2.5x increase)
            'funreq': 70,       # was 30 (2.3x increase)
            'interface': 0,     # Policy makers don't see interfaces
            'solution': 0       # Policy makers don't see standards
        },
        'technical_depth': 'low',
    },
    UserRole.CONSULTANT: {
        'content_types': ['funreq', 'interface', 'spinstance', 'solution', 'plandetail'],
        'response_style': 'technical',
        'max_results_per_type': {
            'plandetail': 40,   # was 15 (2.7x increase)
            'spinstance': 100,  # was 40 (2.5x increase)
            'funreq': 80,       # was 30 (2.7x increase)
            'interface': 50,    # was 20 (2.5x increase)
            'solution': 80      # was 30 (2.7x increase) - Consultants see standards
        },
        'technical_depth': 'high',
    },
    UserRole.MPO_STAFF: {
        'content_types': ['plandetail', 'spinstance', 'funreq', 'solution', 'interface'],
        'response_style': 'technical',
        'max_results_per_type': {
            'plandetail': 100,  # was 40 (2.5x increase)
            'spinstance': 100,  # was 40 (2.5x increase)
            'funreq': 80,       # was 30 (2.7x increase)
            'interface': 60,    # was 25 (2.4x increase)
            'solution': 60      # was 25 (2.4x increase) - MPO staff see standards
        },
        'technical_depth': 'medium-high',
    },
    UserRole.PLANNER: {
        'content_types': ['plandetail', 'spinstance', 'funreq', 'interface'],
        'response_style': 'planning',
        'max_results_per_type': {
            'plandetail': 120,  # was 50 (2.4x increase)
            'spinstance': 100,  # was 40 (2.5x increase)
            'funreq': 80,       # was 30 (2.7x increase)
            'interface': 60,    # was 25 (2.4x increase)
            'solution': 0       # Planners don't see standards
        },
        'technical_depth': 'medium',
    },
    UserRole.ENGINEER: {
        'content_types': ['funreq', 'interface', 'solution', 'spinstance'],
        'response_style': 'technical',
        'max_results_per_type': {
            'plandetail': 100,  # was 40 (2.5x increase)
            'spinstance': 120,  # was 50 (2.4x increase)
            'funreq': 70,       # was 25 (2.8x increase)
            'interface': 100,   # was 40 (2.5x increase)
            'solution': 100     # was 40 (2.5x increase) - Engineers see standards
        },
        'technical_depth': 'high',
    },
    UserRole.UNKNOWN: {
        'content_types': ['plandetail', 'spinstance', 'funreq', 'interface', 'solution'],
        'response_style': 'balanced',
        'max_results_per_type': {
            'plandetail': 60,
            'spinstance': 60,
            'funreq': 50,
            'interface': 40,
            'solution': 40
        },
        'technical_depth': 'medium',
    }
}

# ============================================================================
# SECTION 2: SESSION MANAGEMENT
# ============================================================================

# Session storage: session_id -> session data
sessions: Dict[str, dict] = {}

# Session configuration
MAX_QUERIES_PER_DAY = 10
MAX_QUERIES_PER_CONVERSATION = 3  # Limit continuous conversation to 3 queries
SESSION_CLEANUP_HOURS = 48  # Clean up sessions older than 48 hours

class SessionData(BaseModel):
    """Session data structure"""
    session_id: str
    conversation_history: List[dict]  # [{role: user/assistant, content: str}]
    query_count: int  # Daily query count
    conversation_query_count: int = 0  # Queries in current conversation
    created_at: datetime
    last_activity: datetime
    user_role: Optional[str] = None

def get_midnight_today():
    """Get midnight (00:00:00) for today"""
    now = datetime.now()
    return datetime(now.year, now.month, now.day, 0, 0, 0)

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, dict]:
    """
    Get existing session or create new one.
    Returns: (session_id, session_data)
    """
    # Clean up old sessions first
    cleanup_old_sessions()

    # Check for midnight reset
    reset_midnight_sessions()

    if session_id and session_id in sessions:
        # Return existing session
        sessions[session_id]['last_activity'] = datetime.now()
        return session_id, sessions[session_id]

    # Create new session
    new_session_id = str(uuid.uuid4())
    sessions[new_session_id] = {
        'session_id': new_session_id,
        'conversation_history': [],
        'query_count': 0,
        'conversation_query_count': 0,
        'created_at': datetime.now(),
        'last_activity': datetime.now(),
        'user_role': None
    }

    print(f"Created new session: {new_session_id}")
    return new_session_id, sessions[new_session_id]

def cleanup_old_sessions():
    """Remove sessions older than SESSION_CLEANUP_HOURS"""
    cutoff_time = datetime.now() - timedelta(hours=SESSION_CLEANUP_HOURS)
    old_sessions = [sid for sid, data in sessions.items()
                   if data['last_activity'] < cutoff_time]

    for sid in old_sessions:
        del sessions[sid]
        print(f"Cleaned up old session: {sid}")

def reset_midnight_sessions():
    """Reset query counts for sessions that were created before today's midnight"""
    midnight_today = get_midnight_today()

    for session_id, session_data in sessions.items():
        # If session was created before today's midnight, reset the query count
        if session_data['created_at'] < midnight_today:
            print(f"Resetting session {session_id} - created before midnight")
            session_data['query_count'] = 0
            session_data['created_at'] = datetime.now()  # Reset creation time

def check_query_limit(session_data: dict) -> tuple[bool, int]:
    """
    Check if user has exceeded daily query limit.
    Returns: (can_query, remaining_queries)
    """
    remaining = MAX_QUERIES_PER_DAY - session_data['query_count']
    return remaining > 0, remaining

def check_conversation_limit(session_data: dict) -> tuple[bool, int]:
    """
    Check if user has exceeded conversation query limit.
    Returns: (can_query, remaining_in_conversation)
    """
    conversation_count = session_data.get('conversation_query_count', 0)
    remaining = MAX_QUERIES_PER_CONVERSATION - conversation_count
    return remaining > 0, remaining

# ============================================================================
# SECTION 3: EXISTING CONTENT LOADING
# ============================================================================

content_data = []

def load_content_data():
    global content_data
    try:
        # Load JSONL format (one JSON object per line)
        with open('processed_content.jsonl', 'r', encoding='utf-8') as f:
            content_data = [json.loads(line) for line in f]
        print(f"Loaded {len(content_data)} content chunks from JSONL")

        # Count unique documents
        unique_parents = len(set(item.get('parent_id', 'unknown') for item in content_data))
        print(f"Representing {unique_parents} unique documents")

    except FileNotFoundError:
        print("Error: processed_content.jsonl not found. Run content_processor.py first.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global content_data
    if len(content_data) == 0:
        load_content_data()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini Configuration
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=gemini_api_key)

# ============================================================================
# SECTION 4: PYDANTIC MODELS
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    current_page: Optional[str] = None
    session_id: Optional[str] = None
    role: Optional[str] = None  # User role for content filtering
    conversation_history: Optional[List[dict]] = None  # Previous conversation messages

class ChatResponse(BaseModel):
    response: str
    session_id: str
    remaining_queries: int  # Daily remaining
    query_count: int  # Daily count
    conversation_query_count: int = 0  # Queries in current conversation
    remaining_in_conversation: int = 3  # Remaining in current conversation
    # sources: List[str]

# ============================================================================
# SECTION 5: ROLE DETECTION FUNCTION
# ============================================================================
def get_user_role(role_value: str) -> UserRole:
    """
    Get user role directly from form selection.
    No text parsing needed.
    """
    try:
        return UserRole[role_value]
    except KeyError:
        return UserRole.UNKNOWN

# ============================================================================
# SECTION 6: SEARCH FUNCTIONS (Replace your existing find_relevant_content)
# ============================================================================

def find_relevant_content(query: str, max_results: int = 50, expanded_terms: List[str] = None):
    """
    Keyword-based content search with LLM query expansion support.
    Used as one component of multi-stage retrieval.
    Now filters out stop words and supports expanded search terms.
    """
    scored_content = []

    # Use expanded terms if provided, otherwise fall back to original query
    search_terms = expanded_terms if expanded_terms else [query]

    for item in content_data:
        score = 0
        content_lower = item['content'].lower()
        title_lower = item['title'].lower()

        # Score each expanded term
        for term_index, term in enumerate(search_terms):
            term_words = term.lower().split()
            weight = 3 if term_index == 0 else 1  # Original query gets higher weight

            # Check if this is a phrase (multi-word term)
            if len(term_words) > 1:
                # Phrase matching: bonus for exact phrase match
                if term.lower() in content_lower:
                    score += 5 * weight  # Phrase match in content
                if term.lower() in title_lower:
                    score += 15 * weight  # Phrase match in title (higher weight)

            # ALWAYS do word-level matching (for both single and multi-word terms)
            for word in term_words:
                # Skip very short words and common stop words
                if len(word) > 2 and word not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'would', 'could', 'there', 'their', 'will', 'each', 'make', 'like', 'him', 'into', 'time', 'very', 'when', 'come', 'made', 'find', 'than', 'first', 'been', 'its', 'who', 'now', 'way', 'may', 'down', 'did', 'get', 'come', 'made', 'about', 'want', 'tell'}:
                    score += content_lower.count(word) * 1 * weight
                    score += title_lower.count(word) * 3 * weight

        if score > 0:
            scored_content.append((score, item))

    scored_content.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored_content[:max_results]]


def find_relevant_content_by_type(query: str, url_pattern: str, max_results: int = 50, expanded_terms: List[str] = None):
    """
    Search for content matching a specific URL pattern with LLM query expansion support.
    Now filters out stop words and supports expanded search terms.

    Example: find_relevant_content_by_type(query, 'interface.htm', 3)
             → Returns top 3 interface documents
    """
    scored_content = []

    # Use expanded terms if provided, otherwise fall back to original query
    search_terms = expanded_terms if expanded_terms else [query]

    for item in content_data:
        # Only consider items matching the URL pattern
        if url_pattern not in item['url'].lower():
            continue

        score = 0
        content_lower = item['content'].lower()
        title_lower = item['title'].lower()

        # Score each expanded term
        for term_index, term in enumerate(search_terms):
            term_words = term.lower().split()
            weight = 3 if term_index == 0 else 1  # Original query gets higher weight

            # Check if this is a phrase (multi-word term)
            if len(term_words) > 1:
                # Phrase matching: bonus for exact phrase match
                if term.lower() in content_lower:
                    score += 5 * weight
                if term.lower() in title_lower:
                    score += 15 * weight

            # ALWAYS do word-level matching (for both single and multi-word terms)
            for word in term_words:
                # Skip very short words and common stop words
                if len(word) > 2 and word not in {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'would', 'could', 'there', 'their', 'will', 'each', 'make', 'like', 'him', 'into', 'time', 'very', 'when', 'come', 'made', 'find', 'than', 'first', 'been', 'its', 'who', 'now', 'way', 'may', 'down', 'did', 'get', 'come', 'made', 'about', 'want', 'tell'}:
                    score += content_lower.count(word) * 1 * weight
                    score += title_lower.count(word) * 3 * weight

        if score > 0:
            scored_content.append((score, item))

    scored_content.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in scored_content[:max_results]]


def expand_query_with_llm(query: str) -> List[str]:
    """
    Use Gemini Flash to expand user query with related terms, phrases, acronyms, and synonyms.
    This significantly improves search recall by finding semantically related content.

    Args:
        query: Original user query string

    Returns:
        List of expanded search terms (including original query)
    """
    expansion_prompt = f"""You are a search query expansion expert for an Intelligent Transportation Systems (ITS) architecture database.

Given this user query: "{query}"

Your task: Generate 10-15 related search terms that would help find relevant ITS content.

IMPORTANT: Identify multi-word phrases and keep them together (e.g., "New York", not "New" and "York" separately).

Include:
1. Multi-word phrases from the query (keep phrases intact)
2. Common acronyms (e.g., "DSS" for "decision support system", "NYS" for "New York State")
3. Synonyms and related terms
4. Related ITS concepts and technical terms

Return ONE term per line. No numbering, no bullets, no explanations.

Example for "decision support system for New York":
decision support system
decision support
New York
New York State
NYS
NYC
DSS
decision making
analytical tools
data analysis
operations center
TMC
traffic management

Now generate expanded terms for: "{query}"
"""

    try:
        print(f"\n=== Calling LLM for Query Expansion ===")
        print(f"Model: gemini-3-flash-preview")
        print(f"Query being expanded: {query}")

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": expansion_prompt}]
                }
            ],
            config={
                "system_instruction": "You are a search query expansion expert. Return one search term per line.",
                "max_output_tokens": 2000,
                "temperature": 1.0,
            }
        )

        # DEBUG: Print full response object
        print(f"\n=== DEBUG: Full Response Object ===")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        print(f"Response repr: {repr(response)}")

        # Try to access output_text
        try:
            text = response.output_text.strip()
            print(f"output_text value: '{text}'")
            print(f"output_text length: {len(text)}")
        except AttributeError as ae:
            print(f"AttributeError accessing output_text: {ae}")
            # Try alternative attributes
            if hasattr(response, 'text'):
                text = response.text.strip()
                print(f"Using response.text instead: '{text}'")
            elif hasattr(response, 'content'):
                text = response.content.strip()
                print(f"Using response.content instead: '{text}'")
            else:
                raise
        print("=" * 50)

        expanded_terms = [line.strip() for line in text.split('\n') if line.strip()]

        # Always include the original query as the first term
        if query not in expanded_terms:
            expanded_terms.insert(0, query)

        print(f"\n=== Query Expansion SUCCESS ===")
        print(f"Original: {query}")
        print(f"Expanded to {len(expanded_terms)} terms:")
        for i, term in enumerate(expanded_terms[:10], 1):
            print(f"  {i}. {term}")
        if len(expanded_terms) > 10:
            print(f"  ... and {len(expanded_terms) - 10} more")
        print("================================\n")

        return expanded_terms

    except AttributeError as e:
        print(f"\n!!! Query Expansion ERROR (AttributeError) !!!")
        print(f"Error: {e}")
        print(f"This suggests the API client method or attribute doesn't exist.")
        print(f"Check: client.responses.create() or response.output_text")
        print(f"Falling back to original query only.")
        print("=" * 50 + "\n")
        return [query]

    except Exception as e:
        print(f"\n!!! Query Expansion ERROR (General Exception) !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print(f"Falling back to original query only.")

        # Import traceback for detailed error info
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("=" * 50 + "\n")

        return [query]


def _is_allowed_content_type(url: str, allowed_types: List[str]) -> bool:
    """
    HELPER: Check if a URL matches allowed content types for the user's role.
    
    Example: If user is POLICY_MAKER (allowed_types = ['plandetail', 'spinstance']),
             and url contains 'solution' (standards), this returns False.
    """
    url_lower = url.lower()
    
    type_patterns = {
        'funreq': 'funreq',
        'interface': 'interface.htm',
        'spinstance': 'spinstance',
        'solution': 'solution',
        'plandetail': 'plandetail'
    }
    
    for content_type in allowed_types:
        if content_type in type_patterns:
            if type_patterns[content_type] in url_lower:
                return True
    
    # Allow general content that doesn't match specific patterns
    if not any(pattern in url_lower for pattern in type_patterns.values()):
        return True
    
    return False


def find_relevant_content_multi_stage(query: str, user_role: UserRole):
    """
    Multi-stage retrieval with LLM query expansion and role-based filtering.
    This is the MAIN search function.

    Logical flow:
    1. Expand query using Gemini Flash (identify phrases, acronyms, synonyms)
    2. Get role configuration (what can this user see?)
    3. Stage 1: General search with expanded terms (top 15)
    4. Stage 2: Targeted searches for each allowed content type
    5. Return combined, deduplicated results
    """
    all_results = []
    seen_urls = set()

    # Step 0: Expand query with LLM (NEW!)
    expanded_terms = expand_query_with_llm(query)

    # Get role configuration
    role_config = ROLE_CONTENT_CONFIG[user_role]
    allowed_types = role_config['content_types']
    max_per_type = role_config['max_results_per_type']

    print(f"\n=== Role-Based Retrieval ===")
    print(f"Detected Role: {user_role.value}")
    print(f"Allowed Content Types: {allowed_types}")
    print(f"Max per type: {max_per_type}")
    print("===========================\n")

    # Stage 1: General high-relevance search with expanded terms
    general_results = find_relevant_content(query, max_results=15, expanded_terms=expanded_terms)
    for item in general_results:
        if _is_allowed_content_type(item['url'], allowed_types):
            if item['url'] not in seen_urls:
                all_results.append(item)
                seen_urls.add(item['url'])
    
    # Stage 2: Targeted searches based on role configuration
    query_lower = query.lower()
    
    # Planning documents
    if 'plandetail' in allowed_types and max_per_type.get('plandetail', 0) > 0:
        plan_results = find_relevant_content_by_type(
            query, 'plandetail', max_results=max_per_type['plandetail'], expanded_terms=expanded_terms
        )
        for item in plan_results:
            if item['url'] not in seen_urls:
                all_results.append(item)
                seen_urls.add(item['url'])

    # Service packages
    if 'spinstance' in allowed_types and max_per_type.get('spinstance', 0) > 0:
        sp_results = find_relevant_content_by_type(
            query, 'spinstance', max_results=max_per_type['spinstance'], expanded_terms=expanded_terms
        )
        for item in sp_results:
            if item['url'] not in seen_urls:
                all_results.append(item)
                seen_urls.add(item['url'])

    # Functional requirements
    if 'funreq' in allowed_types and max_per_type.get('funreq', 0) > 0:
        funreq_results = find_relevant_content_by_type(
            query, 'funreq', max_results=max_per_type['funreq'], expanded_terms=expanded_terms
        )
        for item in funreq_results:
            if item['url'] not in seen_urls:
                all_results.append(item)
                seen_urls.add(item['url'])

    # Interfaces (only if role allows)
    if 'interface' in allowed_types and max_per_type.get('interface', 0) > 0:
        interface_results = find_relevant_content_by_type(
            query, 'interface.htm', max_results=max_per_type['interface'], expanded_terms=expanded_terms
        )
        for item in interface_results:
            if item['url'] not in seen_urls:
                all_results.append(item)
                seen_urls.add(item['url'])

    # Standards (ONLY for consultant, MPO, engineer)
    if 'solution' in allowed_types and max_per_type.get('solution', 0) > 0:
        standard_results = find_relevant_content_by_type(
            query, 'solution', max_results=max_per_type['solution'], expanded_terms=expanded_terms
        )
        for item in standard_results:
            if item['url'] not in seen_urls:
                all_results.append(item)
                seen_urls.add(item['url'])
    
    print(f"Total results found: {len(all_results)}")
    return all_results

# ============================================================================
# SECTION 7: ROLE-SPECIFIC SYSTEM PROMPTS (NEW)
# ============================================================================

def build_role_specific_system_prompt(user_role: UserRole) -> str:
    """
    Build a system prompt tailored to the user's role.
    This tells the LLM HOW to format the response based on who's asking.
    """

    base_prompt = """# --- Role and Goal ---
You are an expert assistant for the Intelligent Transportation Systems (ITS) Architecture website. Your primary goal is to help users understand the ITS architecture by providing clear, accurate, and well-formatted answers based *only* on the context provided.

# --- Core Instructions ---
1. **Analyze the User's Question:** Determine the user's intent based on their stated role and what they're likely trying to accomplish.
2. **Synthesize Information:** Combine relevant facts from the different context sources to create a comprehensive, logical answer.
3. **Structure Your Response:** Use markdown for clarity. Use a main heading (##) for the response title, sub-headings (###) for sections, and bullet points (`-` or `*`) for lists. Adapt structure to fit the question—don't force a rigid template.

# --- Source Citation Rules (CRITICAL) ---
- **Cite Per Section, Not Per Sentence:** Cite each unique source ONCE at the end of the section or paragraph that uses it. Do NOT repeat the same citation multiple times within a section.
- **Group Related Citations:** If multiple sentences in a section come from the same source, cite that source once at the end of the section.
- **NO Final Source List:** Do NOT add a separate "## Sources" section at the end of your response.
- **Correct Link Format:** Sources MUST be cited using clickable markdown links in the exact format: `[Title](URL)`.
- **Use Provided Data:** The `Title` and `URL` must come from the context provided.
- **NEVER use file names or paths** (like '../content/bundle1046.htm') in your response.
- **Avoid "Triplet" URLs:** Do NOT use URLs that contain "triplet" in the link.
- **Cite Specific Pages:** Always cite specific Service Packages or Interfaces, not general pages.
- **Limit Citations Per Section:** Each section should have 1-3 citations maximum. Choose the most authoritative/relevant sources.

"""

    # Add role-specific instructions
    if user_role == UserRole.POLICY_MAKER:
        role_instructions = """
# --- AUDIENCE: Policy Maker / Funding Agency ---
Your user is a **POLICY MAKER** or **FUNDING AGENCY** representative.

**Their Likely Tasks:**
- Deciding whether to fund ITS projects
- Understanding strategic value of ITS investments
- Communicating benefits to elected officials or the public
- Evaluating alignment with transportation goals

**What to Include:**
- Strategic value and public benefit statements
- Stakeholder impacts (who benefits, who operates)
- Alignment with regional/state transportation goals
- High-level cost and timeline considerations
- Service packages described in terms of *outcomes* and *public value*, not technical features

**What to EXCLUDE (CRITICAL):**
- NO technical standards (NTCIP, TMDD, etc.) - never mention protocol names
- NO interface specifications or data flows
- NO functional requirement IDs
- NO protocol details or data formats
- NO implementation-level technical details

**Handling Technical Questions:**
If asked about something technical (e.g., "What is NTCIP?"), provide a one-sentence plain-language explanation of what it accomplishes, then redirect to strategic value: "For your purposes, what matters is that this ensures different vendors' equipment works together, protecting your investment."

**Suggested Response Structure:**
1. **Executive Summary** (2-3 sentences of strategic value)
2. **Strategic Value & Benefits** (who benefits, how)
3. **Key Capabilities** (service packages in plain language, max 4-5)
4. **Stakeholders & Coordination** (who operates, who's involved)
5. **Planning Alignment** (connection to strategic goals)

**Length:** Keep responses concise—roughly 60% the length of a technical response.
"""

    elif user_role == UserRole.CONSULTANT:
        role_instructions = """
# --- AUDIENCE: Consultant / System Integrator ---
Your user is a **CONSULTANT** or **SYSTEM INTEGRATOR**.

**Their Likely Tasks:**
- Writing RFPs and system specifications
- Designing system architectures
- Evaluating vendor proposals
- Planning system integrations
- Creating technical documentation

**What to Include:**
- Complete functional requirements WITH specific IDs (e.g., "FR-123")
- Detailed interface specifications with interface IDs
- All applicable ITS standards with specific document references (NTCIP 1203, TMDD, etc.)
- Service packages with implementation details
- Data flows and system interconnections
- Technical constraints and integration considerations

**What to EXCLUDE:**
- Nothing—consultants need comprehensive access
- However, ORGANIZE technical detail logically rather than dumping it

**Response Priority:**
Lead with what's most actionable for RFP/design work:
1. Requirements first (specific, numbered)
2. Then interfaces and data flows
3. Then applicable standards
4. Then service packages for context

**Suggested Response Structure:**
1. **Technical Overview** (scope and purpose)
2. **Functional Requirements** (detailed, with IDs and sources)
3. **Interface Specifications** (data flows, protocols, interface IDs)
4. **Applicable Standards** (specific standard documents)
5. **Service Packages** (implementation context, 5-8 packages)
6. **Implementation Considerations** (constraints, integration points)

**Length:** Provide thorough, detailed responses with extensive citations.
"""

    elif user_role == UserRole.MPO_STAFF:
        role_instructions = """
# --- AUDIENCE: MPO or Transit Agency Staff ---
Your user is **MPO** or **TRANSIT AGENCY STAFF**.

**Their Likely Tasks:**
- Coordinating regional ITS planning across multiple agencies
- Facilitating inter-agency agreements and data sharing
- Ensuring interoperability across jurisdictions
- Connecting ITS projects to regional transportation plans (TIP, LRTP, UPWP)
- Managing regional ITS architecture updates

**What to Include:**
- Multi-agency coordination aspects (this is their primary concern)
- Regional planning document references
- Service packages that span agency boundaries
- Interoperability considerations (explain WHAT standards enable, not protocol details)
- Stakeholder roles and responsibilities across agencies
- Connections to MPO planning processes

**What to EXCLUDE:**
- Deep protocol-level details (mention standards exist for interoperability, don't explain SNMP)
- Individual functional requirement IDs (summarize capabilities instead)
- Vendor-specific implementation details
- Single-agency operational details

**Key Differentiator:**
MPO staff care about REGIONAL scope and INTER-AGENCY coordination. Every response should address: "How does this work across agency boundaries?"

**Suggested Response Structure:**
1. **Overview** (regional coordination context)
2. **Multi-Agency Coordination** (how agencies work together)
3. **Service Packages** (focus on cross-jurisdictional packages, 4-6)
4. **Interoperability Requirements** (what needs to be compatible)
5. **Regional Planning Alignment** (connection to TIP, LRTP, regional goals)
6. **Stakeholder Roles** (which agencies, what responsibilities)

**Length:** Balance planning context with enough technical detail for coordination decisions.
"""

    elif user_role == UserRole.PLANNER:
        role_instructions = """
# --- AUDIENCE: Transportation Planner ---
Your user is a **TRANSPORTATION PLANNER**.

**Their Likely Tasks:**
- Developing TSMO (Transportation Systems Management and Operations) plans
- Writing Concept of Operations (ConOps) documents
- Aligning ITS with agency strategic plans
- Justifying ITS investments in planning documents
- Creating Congestion Management Process documentation

**What to Include:**
- Strategic planning context and goal alignment
- Service packages with planning implications (what capabilities they provide)
- Functional capabilities at a "what does this enable" level (not requirement IDs)
- How ITS supports TSMO objectives
- Phasing and implementation considerations for planning
- Coordination needs relevant to their agency

**What to EXCLUDE:**
- Technical standards details (mention compliance is required, don't detail protocols)
- Interface-level data flows and protocol specifications
- Specific functional requirement IDs (describe capabilities instead)
- Deep technical implementation details

**Key Differentiator:**
Planners focus on THEIR AGENCY'S plan and how ITS fits into it. Less emphasis on inter-agency coordination unless specifically asked. Frame everything through TSMO/planning objectives.

**Suggested Response Structure:**
1. **Overview** (planning perspective and strategic fit)
2. **TSMO/Strategic Alignment** (how this supports planning objectives)
3. **Key Capabilities** (what the system enables, in planning terms)
4. **Service Packages** (planning implications, 4-5 packages)
5. **Implementation Considerations** (phasing, dependencies)
6. **Coordination Needs** (if applicable to their planning scope)

**Length:** Focus on planning implications rather than technical implementation.
"""

    elif user_role == UserRole.ENGINEER:
        role_instructions = """
# --- AUDIENCE: Traffic Operations Engineer ---
Your user is a **TRAFFIC OPERATIONS ENGINEER**.

**Their Likely Tasks:**
- Configuring and deploying ITS systems
- Troubleshooting integration issues
- Ensuring standards compliance
- Understanding data flows for system design
- Operating Traffic Management Centers
- Maintaining field devices

**What to Include:**
- Detailed functional requirements WITH specific IDs
- Interface specifications with data flow details
- Protocol-level standard details (NTCIP objects, TMDD elements, specific versions)
- Technical constraints and performance requirements
- Service packages with technical implementation focus
- System integration and interoperability details

**What to EXCLUDE:**
- Strategic planning context padding (unless directly relevant to the technical question)
- Policy-level justifications
- Lengthy stakeholder coordination discussions
- High-level "strategic value" content

**Response Priority:**
Get to the technical specifics quickly. Engineers don't need strategic context padding—lead with actionable technical information.

**Suggested Response Structure:**
1. **Technical Overview** (brief, then dive into details)
2. **Functional Requirements** (specific IDs, detailed specs)
3. **Data Flows & Interfaces** (protocols, interface IDs, data elements)
4. **Standards Compliance** (specific standards, versions, relevant sections)
5. **Service Packages** (technical implementation details, 4-6)
6. **Technical Considerations** (integration, performance, troubleshooting)

**Length:** Provide deep technical detail with comprehensive citations. Be thorough but organized.
"""

    else:  # UNKNOWN
        role_instructions = """
# --- AUDIENCE: General User ---
Provide a balanced response appropriate for a general audience:

**What to Include:**
- Clear overview of the topic
- Moderate technical depth
- Service packages and their purposes
- Key capabilities and benefits

**What to EXCLUDE:**
- Deep technical standards details
- Protocol-level specifications
- Highly specialized jargon without explanation

**Suggested Response Structure:**
1. **Overview** (what is this and why does it matter)
2. **Key Capabilities** (what it does)
3. **Service Packages** (3-4 relevant packages)
4. **Coordination & Implementation** (brief)

**Length:** Balanced, moderate-length response.
"""

    # Add role-specific example section
    if user_role == UserRole.POLICY_MAKER:
        example_section = """
# --- Example Response for Policy Maker ---
**User Question:** "What are Dynamic Message Signs and should we invest in them?"

**Your Ideal Response:**
## Dynamic Message Signs: Investment Overview

Dynamic Message Signs (DMS) are electronic roadside displays that communicate real-time information to travelers, improving safety and reducing congestion. [Traveler Information Overview](URL)

### Strategic Value
- **Safety Impact**: Reduces secondary crashes by alerting drivers to incidents ahead
- **Congestion Reduction**: Enables route diversion during major incidents, improving traffic flow
- **Public Communication**: Supports Amber Alerts, emergency notifications, and travel time information

### Key Stakeholders
- NYSDOT (ownership and operations)
- Regional Traffic Management Centers (content management)
- Law enforcement (emergency messaging coordination)

### Planning Alignment
This technology supports the Regional ITS Architecture goals for improving traveler information and incident management. Implementation typically involves capital costs for sign hardware plus ongoing communications and maintenance. [Regional Planning Document](URL)

*Note: No technical standards, no interface specs, no requirement IDs—pure strategic/value focus.*
"""

    elif user_role == UserRole.CONSULTANT:
        example_section = """
# --- Example Response for Consultant ---
**User Question:** "What are the technical requirements for implementing Dynamic Message Signs?"

**Your Ideal Response:**
## DMS Technical Implementation Requirements

Dynamic Message Signs require integration with Traffic Management Center systems and compliance with national ITS standards. [Functional Requirements](URL)

### Functional Requirements
- **FR-DMS-001**: Shall display messages up to 3 lines of 18 characters per line [Funreq](URL)
- **FR-DMS-002**: Shall support remote message upload via NTCIP protocol [Funreq](URL)
- **FR-DMS-003**: Shall log all message changes with timestamps for audit purposes [Funreq](URL)
- **FR-DMS-004**: Shall report operational status including lamp failures [Funreq](URL)

### Interface Specifications
- **Interface IF-44-123** (TMC to DMS): Message upload, status monitoring [Interface](URL)
  - Protocol: NTCIP 1203 over STMP/SNMP
  - Key Objects: dmsMessageMultiString, dmsMessageStatus, dmsControlMode

### Applicable Standards
| Standard | Purpose | Key Sections |
|----------|---------|--------------|
| NTCIP 1203 | DMS object definitions | Section 4 (message objects) |
| NTCIP 2304 | Application profile | Center-to-field communications |
| NTCIP 2101 | Point-to-multipoint | Protocol requirements |

[Standards Reference](URL)

### Service Packages
- **TM06 - Traffic Information Dissemination**: Core DMS functionality [SP](URL)
- **TM07 - Regional Traffic Management**: Multi-agency DMS coordination [SP](URL)
- **TM08 - Traffic Incident Management**: Integration with incident detection [SP](URL)

*Note: Specific requirement IDs, interface numbers, standard references, protocol details—everything needed for RFP/design.*
"""

    elif user_role == UserRole.MPO_STAFF:
        example_section = """
# --- Example Response for MPO Staff ---
**User Question:** "How do Dynamic Message Signs support regional coordination?"

**Your Ideal Response:**
## DMS for Regional Coordination

Dynamic Message Signs play a critical role in multi-agency traffic management, particularly during incidents that cross jurisdictional boundaries. [Regional Planning](URL)

### Regional Coordination Context
The Regional ITS Architecture identifies DMS as a key tool for coordinated incident response between NYSDOT, county DOTs, and municipal traffic operations centers. [Planning Document](URL)

### Multi-Agency Service Packages
- **TM07 - Regional Traffic Management**: Enables shared DMS messaging protocols across agency boundaries [SP](URL)
- **TM08 - Traffic Incident Management**: Coordinates DMS messaging with regional incident detection and response [SP](URL)
- **TM06 - Traffic Information Dissemination**: Foundation for consistent traveler information across the region [SP](URL)

### Interoperability Requirements
For effective regional coordination, agencies must:
- Agree on message priority protocols (who can override whom)
- Establish Center-to-Center data sharing agreements
- Use compatible communications standards to ensure system interoperability [Interface](URL)

### Stakeholder Roles
| Agency | Role | Coordination Need |
|--------|------|-------------------|
| NYSDOT | Primary DMS operator on state routes | Regional message coordination |
| County DOTs | Local DMS operations | Data sharing with state TMC |
| MPO | Planning coordination | Architecture consistency |

### Regional Planning Alignment
This supports Regional ITS Architecture Goal 3.2: "Enable seamless traveler information across jurisdictions." [Planning Document](URL)

*Note: Focus on multi-agency, regional scope, interoperability at coordination level—not protocol details.*
"""

    elif user_role == UserRole.PLANNER:
        example_section = """
# --- Example Response for Planner ---
**User Question:** "How do Dynamic Message Signs fit into our TSMO planning?"

**Your Ideal Response:**
## DMS in TSMO Planning Context

Dynamic Message Signs are a foundational TSMO strategy for traveler information, supporting both operational efficiency and safety objectives. [Planning Document](URL)

### TSMO Strategic Alignment
DMS directly supports these common TSMO objectives:
- **Objective: Reduce incident impacts** - Early traveler notification enables route diversion
- **Objective: Improve travel time reliability** - Real-time information helps travelers make informed decisions
- **Objective: Enhance safety** - Advance warning of incidents reduces secondary crashes

### Service Package Connections
- **TM06 - Traffic Information Dissemination**: The primary service package for DMS operations, covering message management and display [SP](URL)
- **TM08 - Traffic Incident Management**: DMS as part of coordinated incident response [SP](URL)
- **PS01 - Emergency Response**: DMS used for evacuation routing and emergency notifications [SP](URL)

### Key Capabilities for Planning
At a planning level, DMS systems provide:
- Real-time travel time and condition display
- Incident and emergency notification capability
- Integration with 511 and mobile apps for message consistency
- Support for work zone and special event management

### Implementation Considerations
When including DMS in your TSMO plan, consider:
- Phased deployment starting with high-volume corridors
- Integration with existing TMC systems
- Maintenance and operations staffing requirements
- Performance measures (message accuracy, display uptime)

[Planning Document](URL)

*Note: Everything framed through TSMO/planning lens—no protocol details, no requirement IDs.*
"""

    elif user_role == UserRole.ENGINEER:
        example_section = """
# --- Example Response for Engineer ---
**User Question:** "What are the data flows and protocols for Dynamic Message Signs?"

**Your Ideal Response:**
## DMS Data Flows and Protocol Specifications

DMS systems communicate with Traffic Management Centers via standardized NTCIP protocols. [Functional Requirements](URL)

### System Architecture
```
TMC Software → [NTCIP 1203/STMP] → Field Network → DMS Controller → Sign Display
TMC Software ← [Status/Diagnostics] ← Field Network ← DMS Controller
```

### Interface Specifications
**Interface IF-44-123: TMC to DMS Communications** [Interface](URL)
- **Outbound (TMC → DMS)**:
  - Message upload: dmsMessageMultiString (OID 1.3.6.1.4.1.1206.4.2.3.5.8.1.3)
  - Activation: dmsMessageActivate (OID 1.3.6.1.4.1.1206.4.2.3.6.3)
  - Control mode: dmsControlMode

- **Inbound (DMS → TMC)**:
  - Status: dmsMessageStatus, dmsCurrentStatus
  - Diagnostics: dmsPixelFailureTableNumRows, dmsLampFailures
  - Environmental: dmsIllumBrightness

### Functional Requirements
- **FR-DMS-001**: Shall accept message uploads via NTCIP 1203 MULTI markup format [Funreq](URL)
- **FR-DMS-002**: Shall report operational status every 60 seconds via SNMP polling [Funreq](URL)
- **FR-DMS-003**: Shall support message queuing with 8 priority levels [Funreq](URL)
- **FR-DMS-004**: Shall automatically adjust brightness based on ambient light sensors [Funreq](URL)

### Standards Compliance
| Standard | Version | Application |
|----------|---------|-------------|
| NTCIP 1203 | v03 | DMS object definitions |
| NTCIP 1201 | v03 | Global objects (time sync, device ID) |
| NTCIP 2101 | v01 | Point-to-multipoint subnet profile |
| NTCIP 2301 | v02 | STMP application profile |

### Service Packages
- **TM06**: Core DMS control and monitoring [SP](URL)
- **TM07**: Center-to-center coordination for regional DMS [SP](URL)

[Standards Reference](URL)

*Note: Protocol details, OIDs, specific object names, data flows—technical specifics for implementation.*
"""

    else:  # UNKNOWN
        example_section = """
# --- Example Response for General User ---
**User Question:** "What are Dynamic Message Signs?"

**Your Ideal Response:**
## Understanding Dynamic Message Signs

Dynamic Message Signs (DMS) are electronic signs along roadways that display real-time information to drivers. [Overview](URL)

### What They Do
- Display current traffic conditions and travel times
- Alert drivers to incidents, construction, or hazards ahead
- Provide emergency information (Amber Alerts, evacuations)
- Show special event or weather-related messages

### Key Service Packages
- **TM06 - Traffic Information Dissemination**: The main service package covering DMS operations [SP](URL)
- **TM08 - Traffic Incident Management**: How DMS works with incident response [SP](URL)

### How They're Managed
DMS are typically controlled from Traffic Management Centers, where operators can update messages remotely based on current conditions. [Planning Document](URL)
"""

    # Add word limit constraint
    word_limit_section = """

# --- CRITICAL CONSTRAINT: Response Completion ---
- Your response MUST be under 1500 words to ensure it completes fully
- Focus on the most relevant information for the user's role
- Be concise but comprehensive
- Prioritize quality over quantity
- Adapt structure to the question—don't force every section if not relevant
- **CRITICAL: Always finish your response with a complete sentence.** Never end mid-word or mid-link.
- If running long, wrap up with a brief conclusion rather than stopping abruptly
"""

    return base_prompt + role_instructions + example_section + word_limit_section


# ============================================================================
# SECTION 8: CHAT ENDPOINT
# ============================================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint with role-based content filtering, response styling, and session management.
    Parses role from the message text sent by the frontend.
    """
    try:
        # Step 1: Get or create session
        session_id, session_data = get_or_create_session(request.session_id)

        # Step 2: Check daily query limit
        can_query, remaining = check_query_limit(session_data)
        if not can_query:
            return ChatResponse(
                response="<p>You have reached your daily limit of 10 queries. Your limit will reset at midnight.</p>",
                session_id=session_id,
                remaining_queries=0,
                query_count=session_data['query_count'],
                conversation_query_count=session_data.get('conversation_query_count', 0),
                remaining_in_conversation=0
            )

        # Step 2b: Check conversation query limit
        can_continue, remaining_in_conv = check_conversation_limit(session_data)
        if not can_continue:
            return ChatResponse(
                response="<p>You have reached your conversation limit. To continue, please start a new chat by clicking the 'Start New Conversation' button.</p>",
                session_id=session_id,
                remaining_queries=remaining,
                query_count=session_data['query_count'],
                conversation_query_count=session_data.get('conversation_query_count', 0),
                remaining_in_conversation=0
            )

        # Step 3: Parse role and actual query from message
        user_role = UserRole.UNKNOWN
        actual_query = request.message

        # Expected format: "Role: ROLE_NAME\nArea of Interest: actual query text"
        if "Role:" in request.message and "Area of Interest:" in request.message:
            lines = request.message.split('\n')
            for line in lines:
                if line.startswith("Role:"):
                    role_value = line.replace("Role:", "").strip()
                    user_role = get_user_role(role_value)
                    # Store role in session (only if valid, not UNKNOWN)
                    if not session_data['user_role'] and user_role != UserRole.UNKNOWN:
                        session_data['user_role'] = user_role.name
                elif line.startswith("Area of Interest:"):
                    actual_query = line.replace("Area of Interest:", "").strip()

        # Use stored role if not provided in this request
        if user_role == UserRole.UNKNOWN and session_data['user_role']:
            user_role = UserRole[session_data['user_role']]

        print(f"\n=== Chat Request ===")
        print(f"Session ID: {session_id}")
        print(f"Daily Query Count: {session_data['query_count'] + 1}/{MAX_QUERIES_PER_DAY}")
        print(f"Conversation Query Count: {session_data.get('conversation_query_count', 0) + 1}/{MAX_QUERIES_PER_CONVERSATION}")
        print(f"Parsed Role: {user_role.value}")
        print(f"Actual Query: {actual_query}")
        print(f"Current Page: {request.current_page}")
        print("====================\n")

        # Track request start time for logging
        start_time = time.time()

        # Step 4: Find relevant content using multi-stage retrieval
        relevant_content = find_relevant_content_multi_stage(
            query=actual_query,
            user_role=user_role
        )

        if not relevant_content:
            return ChatResponse(
                response="<p>I couldn't find relevant information to answer your question. Please try rephrasing or ask about ITS architecture topics.</p>",
                session_id=session_id,
                remaining_queries=remaining,
                query_count=session_data['query_count'],
                conversation_query_count=session_data.get('conversation_query_count', 0),
                remaining_in_conversation=MAX_QUERIES_PER_CONVERSATION - session_data.get('conversation_query_count', 0)
            )

        # Step 5: Build context from relevant content
        context_parts = []
        for item in relevant_content:
            # Limit content length to avoid token limits
            content_preview = item['content'][:4000] + "..." if len(item['content']) > 1500 else item['content']
            # Use URL directly from JSON
            file_url = item['url']

            # Build context with chunk metadata (JSONL format)
            chunk_info = f"Chunk ID: {item.get('chunk_id', 'N/A')}"
            if item.get('chunk_type'):
                chunk_info += f" | Type: {item['chunk_type']}"
            if item.get('chunk_index') is not None:
                chunk_info += f" | Part {item['chunk_index']+1} of {item.get('total_chunks', '?')}"

            context_parts.append(f"{chunk_info}\nURL: {file_url}\nTitle: {item['title']}\nContent: {content_preview}")

        context = "\n\n---\n\n".join(context_parts)

        # Step 6: Generate role-specific system prompt
        system_prompt = build_role_specific_system_prompt(user_role)

        # Step 7: Build messages array with conversation history from session
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history from session (limit to last 8 messages = 4 exchanges)
        if session_data['conversation_history']:
            # Take last 8 messages (4 user + 4 assistant exchanges)
            recent_history = session_data['conversation_history'][-8:]
            messages.extend(recent_history)

        # Add current context and query
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {actual_query}"})

        # Step 8: Call Gemini API
        # max_output_tokens=4000 provides buffer for ~1500 word responses with links/formatting
        # System prompt instructs 1500 word limit to prevent truncation

        # Convert messages to Gemini format
        # Extract system message (first message)
        system_instruction = messages[0]["content"] if messages[0]["role"] == "system" else None

        # Convert remaining messages to Gemini format
        gemini_contents = []
        for msg in messages[1:]:  # Skip system message
            gemini_contents.append({
                "role": "user" if msg["role"] == "user" else "model",  # Gemini uses "model" instead of "assistant"
                "parts": [{"text": msg["content"]}]
            })

        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=gemini_contents,
            config={
                "system_instruction": system_instruction,
                "max_output_tokens": 4000,
                "temperature": 1.0,
            }
        )

        # Step 9: Convert markdown response to HTML
        markdown_content = response.text.strip()
        html_content = markdown.markdown(markdown_content)

        # Step 10: Update session with conversation history and increment query counts
        session_data['conversation_history'].append({
            "role": "user",
            "content": actual_query
        })
        session_data['conversation_history'].append({
            "role": "assistant",
            "content": markdown_content  # Store markdown for future context
        })
        session_data['query_count'] += 1
        session_data['conversation_query_count'] = session_data.get('conversation_query_count', 0) + 1
        session_data['last_activity'] = datetime.now()

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Log the conversation exchange
        try:
            conversation_logger.log_conversation(
                session_id=session_id,
                user_role=user_role.value,
                user_query=actual_query,
                assistant_response=markdown_content,  # Store markdown version for readability
                conversation_context_length=len(session_data['conversation_history']),
                chunks_retrieved=len(relevant_content),
                response_time_ms=response_time_ms
            )
        except Exception as log_error:
            # Don't fail the request if logging fails
            print(f"Warning: Failed to log conversation: {log_error}")

        return ChatResponse(
            response=html_content,
            session_id=session_id,
            remaining_queries=MAX_QUERIES_PER_DAY - session_data['query_count'],
            query_count=session_data['query_count'],
            conversation_query_count=session_data['conversation_query_count'],
            remaining_in_conversation=MAX_QUERIES_PER_CONVERSATION - session_data['conversation_query_count']
        )

    except Exception as e:
        import traceback
        # Log the error message
        print("\n=== Gemini API Error ===")
        print("Error message:", str(e))
        # Log the full stack trace for debugging
        traceback.print_exc()
        print("========================\n")

        # Return the error to the client
        raise HTTPException(status_code=500, detail="Internal Server Error. Check server logs for details.")

class ResetConversationRequest(BaseModel):
    session_id: str
    clear_role: bool = False  # Whether to also clear the user role

class ResetConversationResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    remaining_queries: int
    remaining_in_conversation: int

@app.post("/api/reset-conversation", response_model=ResetConversationResponse)
async def reset_conversation(request: ResetConversationRequest):
    """
    Reset the current conversation without creating a new session.
    Clears conversation history and resets conversation query count.
    Optionally clears the user role if clear_role is True.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = sessions[request.session_id]

    # Reset conversation-specific data
    session_data['conversation_history'] = []
    session_data['conversation_query_count'] = 0
    session_data['last_activity'] = datetime.now()

    # Optionally clear role
    if request.clear_role:
        session_data['user_role'] = None

    print(f"\n=== Conversation Reset ===")
    print(f"Session ID: {request.session_id}")
    print(f"Role cleared: {request.clear_role}")
    print(f"Daily queries used: {session_data['query_count']}/{MAX_QUERIES_PER_DAY}")
    print("==========================\n")

    return ResetConversationResponse(
        success=True,
        message="Conversation reset successfully",
        session_id=request.session_id,
        remaining_queries=MAX_QUERIES_PER_DAY - session_data['query_count'],
        remaining_in_conversation=MAX_QUERIES_PER_CONVERSATION
    )

@app.get("/")
async def root():
    return {"message": "ITS Chat API is running", "content_files_loaded": len(content_data)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "content_files": len(content_data)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)