"""
Prompt Template System for Seed Generation

This module implements the prompt template system with domain-specific variations
for generating diverse CRM tool-calling prompts.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from dataclasses import dataclass
import random
from enum import Enum


class ComplexityLevel(str, Enum):
    """Complexity levels for generated prompts."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class PromptStyle(str, Enum):
    """Different styles of prompts for diversity."""
    DIRECT = "direct"           # "Update lead X to status Y"
    CONVERSATIONAL = "conversational"  # "Hey, can you help me..."
    FORMAL = "formal"           # "Please proceed to update..."
    URGENT = "urgent"           # "Urgently need to..."
    CONTEXTUAL = "contextual"   # "After the meeting, we need to..."


class EdgeCaseType(str, Enum):
    """Types of edge cases for robustness testing."""
    MISSING_INFO = "missing_info"       # Prompt lacks some required info
    AMBIGUOUS = "ambiguous"             # Multiple interpretations possible
    BOUNDARY = "boundary"               # Extreme values or edge of valid range
    TYPO = "typo"                       # Contains realistic typos
    INFORMAL = "informal"               # Very casual/informal language
    MULTI_TOOL = "multi_tool"           # Could require multiple tools


@dataclass
class PromptTemplate:
    """A template for generating prompts."""
    id: str
    tool_name: str
    template: str
    complexity: ComplexityLevel
    style: PromptStyle
    is_edge_case: bool = False
    edge_case_type: Optional[EdgeCaseType] = None
    required_entities: List[str] = None
    optional_entities: List[str] = None

    def __post_init__(self):
        if self.required_entities is None:
            self.required_entities = []
        if self.optional_entities is None:
            self.optional_entities = []


# =============================================================================
# ENTITY GENERATORS - Generate realistic CRM data
# =============================================================================

class EntityGenerator:
    """Generates realistic CRM entities for prompt generation."""

    FIRST_NAMES = [
        "John", "Sarah", "Michael", "Emily", "David", "Jennifer", "Robert", "Lisa",
        "William", "Amanda", "James", "Jessica", "Christopher", "Ashley", "Daniel",
        "Michelle", "Matthew", "Kimberly", "Andrew", "Elizabeth", "Joshua", "Megan",
        "Ryan", "Lauren", "Brandon", "Stephanie", "Kevin", "Nicole", "Brian", "Rachel",
        "Raj", "Priya", "Wei", "Mei", "Carlos", "Maria", "Ahmed", "Fatima", "Yuki", "Hiroshi"
    ]

    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas", "Jackson", "White",
        "Harris", "Martin", "Thompson", "Moore", "Young", "Allen", "King", "Wright",
        "Lee", "Chen", "Patel", "Kumar", "Singh", "Kim", "Tanaka", "Mueller", "Santos"
    ]

    COMPANIES = [
        "Acme Corp", "TechStart Inc", "Global Solutions", "InnovateTech", "DataFlow Systems",
        "CloudNine Software", "Apex Industries", "Quantum Labs", "NextGen Enterprises",
        "PrimeCore Technologies", "BlueSky Analytics", "Vertex Dynamics", "Horizon Group",
        "Catalyst Partners", "Pinnacle Solutions", "Sterling Innovations", "Evergreen Tech",
        "Summit Industries", "Fusion Labs", "Momentum Digital", "BrightPath Consulting",
        "CoreLogic Systems", "Synergy Software", "Atlas Technologies", "Vanguard Solutions"
    ]

    SOURCES = ["Website", "Referral", "LinkedIn", "Trade Show", "Cold Call", "Email Campaign", "Partner"]

    STATUSES = ["New", "Contacted", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]

    JOB_TITLES = [
        "CEO", "CTO", "CFO", "VP of Sales", "VP of Marketing", "VP of Engineering",
        "Director of Operations", "Product Manager", "Engineering Manager", "Sales Manager",
        "Marketing Director", "IT Director", "Procurement Manager", "Business Development Manager",
        "Chief Revenue Officer", "Head of Partnerships", "Senior Buyer", "Account Executive"
    ]

    SALES_REPS = [
        ("SR-101", "Alex Thompson"), ("SR-102", "Jamie Chen"), ("SR-103", "Morgan Davis"),
        ("SR-104", "Casey Williams"), ("SR-105", "Jordan Taylor"), ("SR-106", "Riley Johnson"),
        ("SR-107", "Sam Martinez"), ("SR-108", "Drew Anderson"), ("SR-109", "Pat Garcia"),
        ("SR-110", "Quinn Brown")
    ]

    ACTIVITY_TYPES = ["Call", "Email", "Meeting", "Demo", "Proposal Review", "Contract Discussion"]

    @classmethod
    def generate_name(cls) -> str:
        return f"{random.choice(cls.FIRST_NAMES)} {random.choice(cls.LAST_NAMES)}"

    @classmethod
    def generate_lead_id(cls) -> str:
        return f"LD-{random.randint(1000, 9999)}"

    @classmethod
    def generate_email(cls, name: str, company: str) -> str:
        first, last = name.lower().split()
        company_short = company.lower().split()[0].replace(",", "").replace(".", "")
        patterns = [
            f"{first}.{last}@{company_short}.com",
            f"{first[0]}{last}@{company_short}.com",
            f"{first}@{company_short}.com",
            f"{last}@{company_short}.com"
        ]
        return random.choice(patterns)

    @classmethod
    def generate_phone(cls) -> str:
        area = random.randint(200, 999)
        prefix = random.randint(200, 999)
        line = random.randint(1000, 9999)
        formats = [
            f"({area}) {prefix}-{line}",
            f"{area}-{prefix}-{line}",
            f"+1 {area} {prefix} {line}"
        ]
        return random.choice(formats)

    @classmethod
    def generate_deal_value(cls, complexity: str = "medium") -> int:
        if complexity == "simple":
            return random.randint(5000, 50000)
        elif complexity == "medium":
            return random.randint(25000, 250000)
        else:
            return random.randint(100000, 2000000)

    @classmethod
    def generate_date(cls, future: bool = True) -> str:
        from datetime import datetime, timedelta
        base = datetime.now()
        if future:
            delta = random.randint(1, 30)
            target = base + timedelta(days=delta)
        else:
            delta = random.randint(1, 90)
            target = base - timedelta(days=delta)
        return target.strftime("%Y-%m-%d")

    @classmethod
    def generate_time(cls) -> str:
        hour = random.randint(9, 17)
        minute = random.choice([0, 15, 30, 45])
        return f"{hour:02d}:{minute:02d}"

    @classmethod
    def generate_datetime(cls, future: bool = True) -> str:
        return f"{cls.generate_date(future)}T{cls.generate_time()}"

    @classmethod
    def generate_sales_rep(cls) -> tuple:
        return random.choice(cls.SALES_REPS)

    @classmethod
    def generate_context(cls) -> Dict[str, Any]:
        """Generate a full context with realistic CRM entities."""
        name = cls.generate_name()
        company = random.choice(cls.COMPANIES)

        return {
            "name": name,
            "lead_id": cls.generate_lead_id(),
            "email": cls.generate_email(name, company),
            "company": company,
            "phone": cls.generate_phone(),
            "title": random.choice(cls.JOB_TITLES),
            "source": random.choice(cls.SOURCES),
            "status": random.choice(cls.STATUSES),
            "deal_value": cls.generate_deal_value(),
            "sales_rep": cls.generate_sales_rep(),
            "date": cls.generate_date(),
            "datetime": cls.generate_datetime(),
            "activity_type": random.choice(cls.ACTIVITY_TYPES)
        }


# =============================================================================
# PROMPT TEMPLATES BY TOOL
# =============================================================================

class PromptTemplateSystem:
    """
    System for managing and generating diverse prompts using templates.

    Implements:
    - Template rotation for diversity
    - Multiple complexity levels
    - Edge case generation
    - Domain-specific variations
    """

    def __init__(self):
        self.templates: Dict[str, List[PromptTemplate]] = {}
        self.entity_generator = EntityGenerator()
        self._register_all_templates()

    def _register_all_templates(self):
        """Register all prompt templates."""
        self._register_update_lead_status_templates()
        self._register_search_leads_templates()
        self._register_create_lead_templates()
        self._register_assign_lead_templates()
        self._register_schedule_followup_templates()
        self._register_log_activity_templates()

    def _register_update_lead_status_templates(self):
        """Templates for update_lead_status tool."""
        templates = [
            # Simple - Direct style
            PromptTemplate(
                id="uls_simple_direct_1",
                tool_name="update_lead_status",
                template="Update lead {lead_id} to {status}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "status"]
            ),
            PromptTemplate(
                id="uls_simple_direct_2",
                tool_name="update_lead_status",
                template="Change {name}'s lead status to {status}. Their ID is {lead_id}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "status", "name"]
            ),
            PromptTemplate(
                id="uls_simple_direct_3",
                tool_name="update_lead_status",
                template="Mark lead {lead_id} as {status}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "status"]
            ),

            # Simple - Conversational style
            PromptTemplate(
                id="uls_simple_conv_1",
                tool_name="update_lead_status",
                template="Hey, can you update {name}'s status to {status}? Lead ID is {lead_id}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                required_entities=["lead_id", "status", "name"]
            ),
            PromptTemplate(
                id="uls_simple_conv_2",
                tool_name="update_lead_status",
                template="I need to mark lead {lead_id} as {status}. Thanks!",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                required_entities=["lead_id", "status"]
            ),

            # Medium - Contextual style
            PromptTemplate(
                id="uls_medium_ctx_1",
                tool_name="update_lead_status",
                template="Just finished a call with {name} from {company}. They're interested! Update their lead ({lead_id}) to {status}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "status", "name", "company"]
            ),
            PromptTemplate(
                id="uls_medium_ctx_2",
                tool_name="update_lead_status",
                template="We sent the proposal to {company}. Please change lead {lead_id} ({name}) to {status}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "status", "name", "company"]
            ),
            PromptTemplate(
                id="uls_medium_ctx_3",
                tool_name="update_lead_status",
                template="The meeting with {name} went well. They want to move forward. Update {lead_id} to {status} and add a note: {notes}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "status", "name"],
                optional_entities=["notes"]
            ),

            # Complex - Multi-step context
            PromptTemplate(
                id="uls_complex_1",
                tool_name="update_lead_status",
                template="After several meetings with {name} ({title}) at {company}, we've agreed on terms. The deal is worth ${deal_value}. Update lead {lead_id} to {status}. They're very happy with our proposal.",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "status", "name", "title", "company", "deal_value"]
            ),

            # Urgent style
            PromptTemplate(
                id="uls_urgent_1",
                tool_name="update_lead_status",
                template="URGENT: {name} just called and wants to sign! Quick, update {lead_id} to {status}!",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.URGENT,
                required_entities=["lead_id", "status", "name"]
            ),

            # Edge cases
            PromptTemplate(
                id="uls_edge_ambiguous",
                tool_name="update_lead_status",
                template="The lead with {name} is ready for the next stage. ID: {lead_id}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                is_edge_case=True,
                edge_case_type=EdgeCaseType.AMBIGUOUS,
                required_entities=["lead_id", "name"]
            ),
            PromptTemplate(
                id="uls_edge_informal",
                tool_name="update_lead_status",
                template="yo update {lead_id} 2 {status} pls",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                is_edge_case=True,
                edge_case_type=EdgeCaseType.INFORMAL,
                required_entities=["lead_id", "status"]
            ),
        ]

        self.templates["update_lead_status"] = templates

    def _register_search_leads_templates(self):
        """Templates for search_leads tool."""
        templates = [
            # Simple - Direct
            PromptTemplate(
                id="sl_simple_direct_1",
                tool_name="search_leads",
                template="Find all leads from {company}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["company"]
            ),
            PromptTemplate(
                id="sl_simple_direct_2",
                tool_name="search_leads",
                template="Search for leads with status {status}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["status"]
            ),
            PromptTemplate(
                id="sl_simple_direct_3",
                tool_name="search_leads",
                template="Show me all {status} leads.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["status"]
            ),

            # Simple - Conversational
            PromptTemplate(
                id="sl_simple_conv_1",
                tool_name="search_leads",
                template="Can you find leads assigned to {sales_rep_name}?",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                required_entities=["sales_rep_name"]
            ),
            PromptTemplate(
                id="sl_simple_conv_2",
                tool_name="search_leads",
                template="I'm looking for leads from {source}. Can you help?",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                required_entities=["source"]
            ),

            # Medium - Multiple filters
            PromptTemplate(
                id="sl_medium_1",
                tool_name="search_leads",
                template="Find all {status} leads from {source} assigned to {sales_rep_name}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["status", "source", "sales_rep_name"]
            ),
            PromptTemplate(
                id="sl_medium_2",
                tool_name="search_leads",
                template="Search for leads at {company} that came through {source}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["company", "source"]
            ),
            PromptTemplate(
                id="sl_medium_3",
                tool_name="search_leads",
                template="Show me leads worth more than ${min_value} that are currently {status}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["min_value", "status"]
            ),

            # Complex - Multiple conditions
            PromptTemplate(
                id="sl_complex_1",
                tool_name="search_leads",
                template="I need to find all {status} leads from {source} that are worth between ${min_value} and ${max_value}, and were created after {date}. Limit to {limit} results.",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.FORMAL,
                required_entities=["status", "source", "min_value", "max_value", "date", "limit"]
            ),
            PromptTemplate(
                id="sl_complex_2",
                tool_name="search_leads",
                template="Find me high-value leads (over ${min_value}) from {company} that haven't been contacted yet. I want to see {limit} of them.",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["min_value", "company", "limit"]
            ),

            # Edge case - vague query
            PromptTemplate(
                id="sl_edge_vague",
                tool_name="search_leads",
                template="Find the good leads please.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                is_edge_case=True,
                edge_case_type=EdgeCaseType.AMBIGUOUS,
                required_entities=[]
            ),
        ]

        self.templates["search_leads"] = templates

    def _register_create_lead_templates(self):
        """Templates for create_lead tool."""
        templates = [
            # Simple - Minimal required
            PromptTemplate(
                id="cl_simple_1",
                tool_name="create_lead",
                template="Create a new lead for {name} at {company}. Email: {email}. Source: {source}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["name", "company", "email", "source"]
            ),
            PromptTemplate(
                id="cl_simple_2",
                tool_name="create_lead",
                template="Add {name} ({email}) from {company} as a new lead. They came through {source}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["name", "company", "email", "source"]
            ),

            # Medium - With optional fields
            PromptTemplate(
                id="cl_medium_1",
                tool_name="create_lead",
                template="I met {name} at {source}. They're the {title} at {company}. Create a lead with their email {email} and phone {phone}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["name", "company", "email", "source", "title", "phone"]
            ),
            PromptTemplate(
                id="cl_medium_2",
                tool_name="create_lead",
                template="New lead from {source}: {name} at {company} (email: {email}). Estimated value is ${deal_value}. Priority: {priority}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["name", "company", "email", "source", "deal_value", "priority"]
            ),

            # Complex - Full context
            PromptTemplate(
                id="cl_complex_1",
                tool_name="create_lead",
                template="Just had a great conversation with {name}, the {title} at {company}. Met them at {source}. Email is {email}, phone is {phone}. They're looking at a ${deal_value} deal. High priority, we should move fast. Notes: {notes}",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["name", "company", "email", "source", "title", "phone", "deal_value", "notes"]
            ),

            # Conversational
            PromptTemplate(
                id="cl_conv_1",
                tool_name="create_lead",
                template="Hey, can you add a new lead? It's {name} from {company}, reached us through {source}. Their email is {email}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                required_entities=["name", "company", "email", "source"]
            ),

            # Edge case - missing info
            PromptTemplate(
                id="cl_edge_missing",
                tool_name="create_lead",
                template="Create a lead for {name} from {company}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                is_edge_case=True,
                edge_case_type=EdgeCaseType.MISSING_INFO,
                required_entities=["name", "company"]
            ),
        ]

        self.templates["create_lead"] = templates

    def _register_assign_lead_templates(self):
        """Templates for assign_lead tool."""
        templates = [
            # Simple
            PromptTemplate(
                id="al_simple_1",
                tool_name="assign_lead",
                template="Assign lead {lead_id} to {sales_rep_name} ({sales_rep_id}).",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "sales_rep_id", "sales_rep_name"]
            ),
            PromptTemplate(
                id="al_simple_2",
                tool_name="assign_lead",
                template="Give {lead_id} to {sales_rep_name}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "sales_rep_name", "sales_rep_id"]
            ),

            # Medium - With context
            PromptTemplate(
                id="al_medium_1",
                tool_name="assign_lead",
                template="The lead from {company} ({lead_id}) should go to {sales_rep_name} because {reason}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "company", "sales_rep_name", "sales_rep_id", "reason"]
            ),
            PromptTemplate(
                id="al_medium_2",
                tool_name="assign_lead",
                template="Reassign {lead_id} from {name} to {sales_rep_name} ({sales_rep_id}). Don't notify the new rep.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "name", "sales_rep_name", "sales_rep_id"]
            ),

            # Complex
            PromptTemplate(
                id="al_complex_1",
                tool_name="assign_lead",
                template="{name}'s lead ({lead_id}) at {company} needs to be transferred to {sales_rep_name} ({sales_rep_id}) because {reason}. The deal is worth ${deal_value} and we need to close this quarter.",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "name", "company", "sales_rep_name", "sales_rep_id", "reason", "deal_value"]
            ),

            # Conversational
            PromptTemplate(
                id="al_conv_1",
                tool_name="assign_lead",
                template="Can you please assign lead {lead_id} to {sales_rep_name}? Thanks!",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.CONVERSATIONAL,
                required_entities=["lead_id", "sales_rep_name", "sales_rep_id"]
            ),
        ]

        self.templates["assign_lead"] = templates

    def _register_schedule_followup_templates(self):
        """Templates for schedule_followup tool."""
        templates = [
            # Simple
            PromptTemplate(
                id="sf_simple_1",
                tool_name="schedule_followup",
                template="Schedule a {activity_type} with lead {lead_id} on {datetime}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "activity_type", "datetime"]
            ),
            PromptTemplate(
                id="sf_simple_2",
                tool_name="schedule_followup",
                template="Set up a follow-up {activity_type} for {lead_id} on {date} at {time}.",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "activity_type", "date", "time"]
            ),

            # Medium
            PromptTemplate(
                id="sf_medium_1",
                tool_name="schedule_followup",
                template="Schedule a {duration}-minute {activity_type} with {name} ({lead_id}) on {datetime}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "name", "activity_type", "datetime", "duration"]
            ),
            PromptTemplate(
                id="sf_medium_2",
                tool_name="schedule_followup",
                template="I need to follow up with {name} from {company}. Schedule a {activity_type} on {datetime}. Lead ID is {lead_id}. Remind me {reminder} minutes before.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "name", "company", "activity_type", "datetime", "reminder"]
            ),

            # Complex
            PromptTemplate(
                id="sf_complex_1",
                tool_name="schedule_followup",
                template="Set up a {activity_type} with {name} ({title}) from {company} on {datetime}. The lead ID is {lead_id}. Block {duration} minutes. Notes: {notes}. Send a reminder {reminder} minutes before.",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.FORMAL,
                required_entities=["lead_id", "name", "title", "company", "activity_type", "datetime", "duration", "notes", "reminder"]
            ),
        ]

        self.templates["schedule_followup"] = templates

    def _register_log_activity_templates(self):
        """Templates for log_activity tool."""
        templates = [
            # Simple
            PromptTemplate(
                id="la_simple_1",
                tool_name="log_activity",
                template="Log a {activity_type} with lead {lead_id}. {description}",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "activity_type", "description"]
            ),
            PromptTemplate(
                id="la_simple_2",
                tool_name="log_activity",
                template="Record that I had a {activity_type} with {lead_id}. Notes: {description}",
                complexity=ComplexityLevel.SIMPLE,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "activity_type", "description"]
            ),

            # Medium
            PromptTemplate(
                id="la_medium_1",
                tool_name="log_activity",
                template="Just finished a {duration}-minute {activity_type} with {name} ({lead_id}). {description}. Outcome: {outcome}.",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "name", "activity_type", "description", "duration", "outcome"]
            ),
            PromptTemplate(
                id="la_medium_2",
                tool_name="log_activity",
                template="Log activity for {name} at {company} (lead {lead_id}): Had a {activity_type}. {description}",
                complexity=ComplexityLevel.MEDIUM,
                style=PromptStyle.DIRECT,
                required_entities=["lead_id", "name", "company", "activity_type", "description"]
            ),

            # Complex
            PromptTemplate(
                id="la_complex_1",
                tool_name="log_activity",
                template="I had an important {activity_type} with {name} ({title}) from {company}. Lead ID: {lead_id}. We discussed their requirements and they seemed interested. The call lasted about {duration} minutes. {description}. Overall outcome was {outcome}.",
                complexity=ComplexityLevel.COMPLEX,
                style=PromptStyle.CONTEXTUAL,
                required_entities=["lead_id", "name", "title", "company", "activity_type", "description", "duration", "outcome"]
            ),
        ]

        self.templates["log_activity"] = templates

    def get_templates(self, tool_name: str) -> List[PromptTemplate]:
        """Get all templates for a specific tool."""
        return self.templates.get(tool_name, [])

    def get_templates_by_complexity(
        self,
        tool_name: str,
        complexity: ComplexityLevel
    ) -> List[PromptTemplate]:
        """Get templates filtered by complexity."""
        return [
            t for t in self.get_templates(tool_name)
            if t.complexity == complexity
        ]

    def get_templates_by_style(
        self,
        tool_name: str,
        style: PromptStyle
    ) -> List[PromptTemplate]:
        """Get templates filtered by style."""
        return [
            t for t in self.get_templates(tool_name)
            if t.style == style
        ]

    def get_edge_case_templates(self, tool_name: str) -> List[PromptTemplate]:
        """Get edge case templates for a tool."""
        return [t for t in self.get_templates(tool_name) if t.is_edge_case]

    def sample_template(
        self,
        tool_name: str,
        complexity: Optional[ComplexityLevel] = None,
        style: Optional[PromptStyle] = None,
        edge_case: bool = False
    ) -> Optional[PromptTemplate]:
        """Sample a random template with optional filters."""
        templates = self.get_templates(tool_name)

        if not templates:
            return None

        if complexity:
            templates = [t for t in templates if t.complexity == complexity]
        if style:
            templates = [t for t in templates if t.style == style]
        if edge_case:
            templates = [t for t in templates if t.is_edge_case]
        else:
            templates = [t for t in templates if not t.is_edge_case]

        return random.choice(templates) if templates else None

    def fill_template(
        self,
        template: PromptTemplate,
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """
        Fill a template with entity values.

        Returns:
            Tuple of (filled_prompt, entities_used)
        """
        if context is None:
            context = EntityGenerator.generate_context()

        # Add sales rep info
        if "sales_rep" in str(template.template):
            rep_id, rep_name = context.get("sales_rep", EntityGenerator.generate_sales_rep())
            context["sales_rep_id"] = rep_id
            context["sales_rep_name"] = rep_name

        # Add derived values
        if "min_value" not in context:
            context["min_value"] = random.randint(10000, 50000)
        if "max_value" not in context:
            context["max_value"] = context.get("min_value", 50000) + random.randint(50000, 200000)
        if "limit" not in context:
            context["limit"] = random.choice([5, 10, 20, 25, 50])
        if "duration" not in context:
            context["duration"] = random.choice([15, 30, 45, 60, 90])
        if "reminder" not in context:
            context["reminder"] = random.choice([5, 10, 15, 30])
        if "notes" not in context:
            notes_options = [
                "Great conversation, very interested in our solution.",
                "Need to follow up with more technical details.",
                "Decision maker, fast-moving deal.",
                "Budget approved for Q1.",
                "Interested but evaluating competitors.",
                "Hot lead, move fast!",
                "Requested pricing proposal."
            ]
            context["notes"] = random.choice(notes_options)
        if "reason" not in context:
            reason_options = [
                "better geographic coverage",
                "they have experience with this industry",
                "previous relationship with the company",
                "expertise in enterprise deals",
                "availability and capacity",
                "language skills match"
            ]
            context["reason"] = random.choice(reason_options)
        if "description" not in context:
            desc_options = [
                "Discussed their needs and timeline.",
                "They're interested in our enterprise plan.",
                "Went over the proposal details.",
                "Answered their technical questions.",
                "They want a demo next week.",
                "Positive conversation, moving forward.",
                "Need to address some concerns about implementation."
            ]
            context["description"] = random.choice(desc_options)
        if "outcome" not in context:
            context["outcome"] = random.choice(["Positive", "Neutral", "Negative", "No Response"])
        if "priority" not in context:
            context["priority"] = random.choice(["Low", "Medium", "High", "Critical"])
        if "time" not in context:
            context["time"] = EntityGenerator.generate_time()

        # Fill the template
        prompt = template.template
        entities_used: Dict[str, Any] = {}

        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
                entities_used[key] = value

        return prompt, entities_used

    def generate_prompt(
        self,
        tool_name: str,
        complexity: Optional[ComplexityLevel] = None,
        style: Optional[PromptStyle] = None,
        edge_case: bool = False,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[tuple[str, Dict[str, Any], PromptTemplate]]:
        """
        Generate a complete prompt from templates.

        Returns:
            Tuple of (prompt_text, entities_used, template_used) or None
        """
        template = self.sample_template(tool_name, complexity, style, edge_case)
        if not template:
            return None

        prompt, entities = self.fill_template(template, context)
        return prompt, entities, template

    def get_all_tool_names(self) -> List[str]:
        """Get list of all registered tool names."""
        return list(self.templates.keys())

    def get_template_stats(self) -> Dict[str, Any]:
        """Get statistics about registered templates."""
        stats = {}
        for tool_name, templates in self.templates.items():
            stats[tool_name] = {
                "total": len(templates),
                "by_complexity": {
                    "simple": len([t for t in templates if t.complexity == ComplexityLevel.SIMPLE]),
                    "medium": len([t for t in templates if t.complexity == ComplexityLevel.MEDIUM]),
                    "complex": len([t for t in templates if t.complexity == ComplexityLevel.COMPLEX]),
                },
                "edge_cases": len([t for t in templates if t.is_edge_case]),
                "by_style": {}
            }
            for style in PromptStyle:
                count = len([t for t in templates if t.style == style])
                if count > 0:
                    stats[tool_name]["by_style"][style.value] = count
        return stats
