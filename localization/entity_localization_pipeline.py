import json
import os
import logging
import re
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from moatless.benchmark.utils import get_moatless_instance
from plugins.location_tools.repo_ops.repo_ops import (
    set_current_issue,
    reset_current_issue,
    search_code_snippets,
    get_graph,
    get_graph_entity_searcher,
    get_graph_dependency_searcher,
)
from dependency_graph.build_graph import (
    NODE_TYPE_FILE, NODE_TYPE_CLASS, NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS, EDGE_TYPE_IMPORTS, EDGE_TYPE_INVOKES, EDGE_TYPE_INHERITS
)
from util.utils import convert_to_json
from entity_embedding import LocalizationChainEmbedding

# Extract the prompt template of the entity from the context (reuse ENTITY_EXTRACTION_PROMPT in entity_search_scorer.py)
ENTITY_EXTRACTION_PROMPT = """
You are a code analysis expert. Given an issue description, your task is to identify the most relevant code entities (classes, methods, functions, variables) that are likely involved in the issue.

⚠️ Important: Only extract entities that are explicitly mentioned or strongly implied by the issue description. Do not invent names that are not referenced in the text.

**Issue Description:**
{issue_description}

**Instructions:**
1. Analyze the issue description to identify:
   - **Classes**: e.g., `UserAuthenticator`, `PaymentProcessor`
   - **Methods/Functions**: e.g., `validate_credentials()`, `process_payment()`
   - **Variables/Parameters**: e.g., `user_id`, `transaction_amount`
   - **Error Types/Exceptions**: e.g., `RateLimitExceededError`, `DatabaseConnectionError`
2. **Focus on direct mentions**: Only include entities that are clearly referenced in the issue.
3. **Avoid redundancy**: If multiple terms refer to the same entity (e.g., "the payment handler" and `PaymentProcessor`), pick the most precise name.
4. **Prioritize key components**: Rank entities by how central they are to the issue.
5. **Return only names**: Do not include paths, modules, or extra descriptions.
6. **Limit to {max_entities} entities**: Select only the {max_entities} most relevant and important entities for this issue.

**Output Format:**
Return a JSON list of exactly {max_entities} entity names in order of relevance (most relevant first):
["entity_name1", "entity_name2", "entity_name3", ...]

**Examples:**

1. **Issue Description:**
    Query syntax error with condition and distinct combination
    Description:
    A Count annotation containing both a Case condition and a distinct=True param produces a query error on Django 2.2 (whatever the db backend). A space is missing at least (... COUNT(DISTINCTCASE WHEN ...).

   **Output (if max_entities=3):**
   ["Count", "DISTINCTCASE", "distinct"]

2. **Issue Description:**
   "After upgrading to v2.0, the `UserSession` class sometimes fails to store session data in Redis, causing login loops."

   **Output (if max_entities=2):**
   ["UserSession", "Redis"]

3. **Issue Description:**
   "The `calculate_discount()` function applies incorrect discounts for bulk orders when `customer_type = 'wholesale'`."

   **Output (if max_entities=3):**
   ["calculate_discount", "customer_type", "wholesale"]

Note: Return only the simple names like "__iter__", "page_range", "MyClass", "my_function", etc. Do not include file paths or full qualified names.
Return exactly {max_entities} entities, prioritizing the most important ones if there are more candidates.
"""

# A prompt template that extracts the relevant entity from a code snippet
CODE_SNIPPET_ENTITY_EXTRACTION_PROMPT = """
Based on the following code snippets and problem statement, identify the 4 most relevant entities (files, classes, or functions) that are likely involved in solving this issue.

**Problem Statement:**
{problem_statement}

**Code Snippets:**
{code_snippets}

**Instructions:**
1. Analyze the problem statement to understand what needs to be fixed/implemented
2. Review the code snippets to identify relevant entities
3. **PRIORITIZE DIVERSITY**: Select entities from different files whenever possible to ensure comprehensive coverage
4. **BALANCE RELEVANCE AND DIVERSITY**: Choose entities that are both highly relevant to the issue AND come from different modules/files
5. Avoid selecting multiple entities from the same file unless absolutely necessary
6. Select exactly 4 entities that collectively provide the best coverage for solving the issue
7. For each entity, provide the exact entity ID in the format expected by the codebase

**Selection Strategy:**
- First priority: High relevance to the problem + Different file locations
- Second priority: High relevance to the problem (even if some files overlap)
- Ensure the selected entities represent different aspects or layers of the solution

**Output Format:**
Return a JSON list containing exactly 4 entities, each with the following format:
```json
[
    {{
        "entity_id": "file_path:QualifiedName or just file_path",
        "entity_type": "file|class|function", 
        "relevance_reason": "Brief explanation of why this entity is relevant to the issue",
        "diversity_value": "How this entity adds diversity (e.g., 'different file', 'different layer', 'different functionality')"
    }}
]
```

**Example:**
```json
[
    {{
        "entity_id": "src/models.py:UserModel",
        "entity_type": "class",
        "relevance_reason": "Contains user-related functionality mentioned in the issue",
        "diversity_value": "Model layer from different file"
    }},
    {{
        "entity_id": "src/views.py:UserView",
        "entity_type": "class", 
        "relevance_reason": "Handles user interface logic that may need modification",
        "diversity_value": "View layer from different file"
    }},
    {{
        "entity_id": "src/utils/validators.py:validate_user_input",
        "entity_type": "function",
        "relevance_reason": "Input validation logic relevant to the user issue",
        "diversity_value": "Utility function from different module"
    }},
    {{
        "entity_id": "src/config.py",
        "entity_type": "file",
        "relevance_reason": "Configuration settings that may affect user behavior",
        "diversity_value": "Configuration file from different location"
    }}
]
```

**Remember**: Maximize both relevance to the issue AND diversity across different files/modules to ensure comprehensive localization chain generation.
"""

# Add prompt template for neighbor pre-screening
NEIGHBOR_PREFILTERING_PROMPT = """
You are a code analysis expert helping to select the most relevant and diverse neighbors for exploring a dependency graph to solve a specific issue.

**Issue Description:**
{issue_description}

**Current Entity:** {current_entity}
**Current Entity Type:** {current_entity_type}
**Traversal Depth:** {depth}

**Available Neighbor Entities ({total_count} total):**
{neighbor_list}

**Your Task:**
From the {total_count} available neighbors, select up to {max_selection} most relevant and diverse entities that would be most promising to explore next.

**Selection Criteria:**
1. **Relevance to Issue**: How likely is this neighbor to contain code related to solving the issue?
2. **Diversity**: Avoid selecting too many entities from the same file or with similar names
3. **Strategic Value**: Prioritize entities that could lead to discovering the root cause or solution
4. **Entity Type Variety**: Balance between files, classes, and functions when possible

**Instructions:**
1. Analyze each neighbor entity ID to understand what it likely represents
2. Consider file paths, entity names, and types to assess relevance
3. Ensure diversity by avoiding redundant selections from the same file/module
4. Select entities that complement each other in exploring different aspects of the issue
5. Return exactly the entity IDs that should be explored further (up to {max_selection})

**Output Format:**
Return a JSON object with your selection:
```json
{{
    "selected_neighbors": [
        "neighbor_entity_id_1",
        "neighbor_entity_id_2", 
        ...
    ],
    "selection_reasoning": "Brief explanation of your selection strategy and why these neighbors were chosen",
    "diversity_considerations": "How you ensured diversity in your selection"
}}
```

Focus on strategic exploration that maximizes the chance of finding issue-relevant code while maintaining diversity.
"""

# Node selection prompt template
NODE_SELECTION_PROMPT = """
You are a code analysis expert helping to navigate a dependency graph to solve a specific issue. Given the current context and available neighboring nodes, determine which node would be most promising to explore next.

**Issue Description:**
{issue_description}

**Current Entity:** {current_entity}
**Current Entity Type:** {current_entity_type}
**Traversal Depth:** {depth}

**Available Neighbor Nodes:**
{neighbor_info}

**Context:**
- We are performing graph traversal to find code locations relevant to solving this issue
- Each neighbor represents a related code entity (file, class, or function)
- We need to select the most promising node to continue exploration

**Instructions:**
1. Analyze how each neighbor might relate to solving the issue
2. Consider the traversal depth and whether we should continue or stop
3. Evaluate which neighbor is most likely to contain relevant code for the solution
4. Return your decision on whether to continue exploration and which neighbor to select

**Output Format:**
Return a JSON object with your decision:
```json
{{
    "should_continue": true/false,
    "selected_neighbor": "neighbor_entity_id or null",
    "reasoning": "Explanation of your decision",
    "confidence": 0-100
}}
```

If should_continue is false, set selected_neighbor to null.
If should_continue is true, select the most promising neighbor_entity_id.
"""

# Add prompt template for agent voting
CHAIN_VOTING_PROMPT = """
You are an expert software engineer tasked with identifying the optimal modification location for solving a specific software issue.

**Issue Description:**
{issue_description}

**Available Localization Chains:**
{chains_info}

**Your Task:**
Analyze each localization chain as a potential modification target and vote for the ONE chain where making changes would most likely resolve the issue described above.

**Evaluation Criteria:**
1. **Problem Location Accuracy**: Does this chain contain the actual location where the bug/issue manifests?
2. **Modification Impact**: How directly would changes to this code path affect the described problem?
3. **Code Modifiability**: Is the code in this chain well-structured and safe to modify?
4. **Solution Completeness**: Would fixing this chain likely resolve the entire issue, not just symptoms?
5. **Risk Assessment**: What are the risks of modifying this particular code path?

**Key Questions to Consider:**
- Which chain contains the root cause rather than just related functionality?
- Where would a developer most likely need to make changes to fix this specific issue?
- Which code path, when modified, would have the most direct impact on resolving the problem?
- Which chain provides the clearest entry point for implementing a fix?

**Instructions:**
1. For each chain, analyze whether modifying its code would directly address the issue
2. Consider the logical flow: which chain is most likely to contain the problematic code?
3. Evaluate implementation feasibility: which chain would be safest and most effective to modify?
4. Vote for exactly ONE chain that represents the best modification target
5. Focus on where to make changes, not just what's related to the issue

**Output Format:**
Return a JSON object with your vote:
```json
{{
    "voted_chain_id": "chain_X",
    "confidence": 85,
    "reasoning": "Detailed explanation of why this chain is the best modification target for solving the issue",
    "modification_strategy": "Brief description of what type of changes would be needed in this chain",
    "chain_analysis": {{
        "chain_1": "Assessment of this chain as a modification target",
        "chain_2": "Assessment of this chain as a modification target",
        ...
    }}
}}
```

**Example:**
```json
{{
    "voted_chain_id": "chain_2",
    "confidence": 88,
    "reasoning": "Chain 2 contains the pagination iterator __iter__ method which is where the infinite loop issue described in the problem statement actually occurs. Modifying the logic in this method to properly handle the iteration termination would directly solve the reported bug.",
    "modification_strategy": "Add proper boundary checking and iteration termination logic in the __iter__ method",
    "chain_analysis": {{
        "chain_1": "Contains utility functions but modifications here would not address the core iteration logic issue",
        "chain_2": "Contains the actual iterator implementation where the bug manifests - ideal modification target",
        "chain_3": "Related display logic but changes here would not fix the underlying iteration problem"
    }}
}}
```

Remember: Focus on identifying where code changes should be made to fix the issue, not just which code is conceptually related.
"""

# Add the prompt template for the first round of modification position judgment
MODIFICATION_LOCATION_PROMPT = """
You are an expert software engineer tasked with identifying specific code locations that need to be modified to solve a given issue.

**Issue Description:**
{issue_description}

**Selected Localization Chain:**
{chain_info}

**Your Task:**
Analyze the localization chain and identify the specific locations within this chain that need to be modified to solve the issue. Focus on pinpointing the exact functions, methods, or code blocks that require changes.

**CRITICAL REQUIREMENT FOR INSTRUCTIONS:**
- Each suggested_approach must be a DETAILED, STEP-BY-STEP instruction
- Include specific code examples, parameter names, and implementation details
- Specify exact lines to modify, functions to add, and variables to change
- Provide concrete implementation guidance that a developer can directly follow
- Include error handling, edge cases, and validation requirements
- Mention specific imports, dependencies, or setup needed

**Instructions:**
1. Examine each entity in the localization chain and its code
2. Identify which specific parts of the code are causing the issue or need enhancement
3. Determine the precise locations where modifications should be made
4. Explain why each location needs modification and what type of change is required
5. Prioritize the modifications by importance (most critical first)
6. For each modification, provide DETAILED implementation instructions with specific code examples

**Output Format:**
Return a JSON object with your analysis:
```json
{{
    "modification_locations": [
        {{
            "entity_id": "specific_entity_id",
            "location_description": "Specific function/method/lines that need modification",
            "modification_type": "fix_bug|add_feature|refactor|optimize",
            "priority": "high|medium|low",
            "reasoning": "Detailed explanation of why this location needs modification",
            "suggested_approach": "DETAILED step-by-step implementation instructions with specific code examples, parameter names, exact function signatures, error handling, and complete implementation guidance that can be directly executed by a developer"
        }}
    ],
    "overall_strategy": "Overall approach to solving the issue using these modifications",
    "confidence": 85
}}
```

**Example of DETAILED suggested_approach:**
Instead of: "Add proper termination condition"
Provide: "Modify the __iter__ method in the Paginator class by adding a counter variable 'current_page = 1' at the beginning. Then add a while loop condition 'while current_page <= self.num_pages:' to replace the infinite loop. Inside the loop, yield 'self.page(current_page)' and increment 'current_page += 1'. Add try-catch block to handle PageNotAnInteger and EmptyPage exceptions by catching them and breaking the loop. Import the exceptions 'from django.core.paginator import PageNotAnInteger, EmptyPage' at the top of the file."
"""

# Add prompt template for the second round of comprehensive judgment
COMPREHENSIVE_MODIFICATION_PROMPT = """
You are an expert software engineer participating in a collaborative code review process to determine the best approach for solving a software issue.

**Issue Description:**
{issue_description}

**Selected Localization Chain:**
{chain_info}

**Your Initial Analysis:**
{your_initial_analysis}

**Other Agents' Analyses:**
{other_agents_analyses}

**Your Task:**
Based on the issue, the localization chain, your initial analysis, and insights from other agents, provide a refined and comprehensive analysis of where and how the code should be modified.

**CRITICAL REQUIREMENT FOR REFINED INSTRUCTIONS:**
- Each suggested_approach must be EXTREMELY DETAILED with complete implementation guidance
- Include specific code snippets, exact function signatures, and parameter details
- Provide line-by-line modification instructions where applicable
- Specify all necessary imports, dependencies, and setup requirements
- Include comprehensive error handling and edge case considerations
- Mention testing requirements and validation steps
- Provide specific examples of input/output or before/after code states

**Instructions:**
1. Review your initial analysis and the analyses from other agents
2. Identify common patterns and disagreements in the proposed modifications
3. Synthesize the best insights from all analyses
4. Refine your modification recommendations based on collective wisdom
5. Provide a more comprehensive and well-reasoned final recommendation
6. Ensure each suggested_approach contains exhaustive implementation details

**Output Format:**
Return a JSON object with your refined analysis:
```json
{{
    "refined_modification_locations": [
        {{
            "entity_id": "specific_entity_id",
            "location_description": "Specific function/method/lines that need modification",
            "modification_type": "fix_bug|add_feature|refactor|optimize",
            "priority": "high|medium|low",
            "reasoning": "Enhanced reasoning incorporating insights from other agents",
            "suggested_approach": "EXHAUSTIVE step-by-step implementation guide including: exact code snippets to add/modify/remove, complete function signatures, all required imports, parameter validation, error handling, edge cases, testing considerations, and specific examples of before/after states",
            "supporting_evidence": "References to other agents' insights that support this decision"
        }}
    ],
    "overall_strategy": "Comprehensive strategy refined through collaborative analysis",
    "confidence": 90,
    "key_insights_learned": "What you learned from other agents' analyses",
    "potential_risks": "Potential risks or challenges identified through collaborative review"
}}
```

Remember: Each suggested_approach should be so detailed that a developer can implement it without additional research or clarification.
"""

FINAL_DISCRIMINATOR_PROMPT = """
You are the lead software architect making the final decision on a code modification plan. Multiple expert engineers have provided their analyses for solving a software issue.

**Issue Description:**
{issue_description}

**Selected Localization Chain:**
{chain_info}

**All Agents' Final Analyses:**
{all_agents_analyses}

**Your Task:**
Synthesize all the expert analyses and create a definitive, actionable modification plan that will solve the issue effectively and safely.

**CRITICAL REQUIREMENTS FOR INSTRUCTIONS:**
- Every instruction MUST be a concrete modification action (Add, Remove, Modify, Replace, Insert, etc.)
- NO verification, checking, or validation instructions (avoid "Verify", "Ensure", "Check", "Maintain", etc.)
- Each instruction should specify exactly WHAT to change and HOW to change it
- Focus on direct code modifications that implement the solution

**Instructions:**
1. Analyze all the expert recommendations and identify the most reliable and consistent suggestions
2. Resolve any conflicts between different expert opinions using technical merit
3. Create a prioritized, step-by-step modification plan with ONLY concrete modification actions
4. Ensure the plan is practical, safe, and addresses the root cause of the issue
5. Include specific instructions for each modification
6. The output context should be as detailed as possible
7. Use action verbs like: "Add", "Modify", "Replace", "Insert", "Update", "Change", "Remove", "Implement"

**Output Format:**
Return a comprehensive modification plan:
```json
{{
    "final_plan": {{
        "summary": "High-level summary of the modification approach",
        "modifications": [
            {{
                "step": 1,
                "instruction": "Concrete modification instruction using action verbs (Add/Modify/Replace/etc.)",
                "context": "File path and specific location (e.g., function, method, line range)",
                "type": "fix_bug|add_feature|refactor|optimize",
                "priority": "critical|high|medium|low",
                "rationale": "Why this modification is necessary and how it contributes to solving the issue",
                "implementation_notes": "Specific technical details for implementation"
            }}
        ],
        "execution_order": "The recommended order for implementing these modifications",
        "testing_recommendations": "Suggested testing approach for validating the modifications",
        "risk_assessment": "Potential risks and mitigation strategies"
    }},
    "confidence": 95,
    "expert_consensus": "Summary of areas where experts agreed",
    "resolved_conflicts": "How conflicting expert opinions were resolved"
}}
```

**Examples of GOOD instructions:**
- "Add maxlength attribute to the widget configuration"
- "Modify the widget_attrs method to include max_length parameter"
- "Replace the current field initialization with max_length support"
- "Insert validation logic for maximum length"

**Examples of BAD instructions (DO NOT USE):**
- "Verify the max_length setting" 
- "Ensure proper validation"
- "Check if the field is configured correctly"
- "Maintain the existing functionality"

Focus on creating a plan that can be directly executed by a modification agent with clear, actionable steps.
"""


class EntityLocalizationPipeline:
    """
    A pipeline that performs entity-based localization using graph traversal.

    The pipeline consists of three main stages:
    1. Extract initial entities from context
    2. For each initial entity, search code snippets and extract 4 related entities
    3. Generate localization chains for each related entity, grouped by initial entity
    """

    def __init__(self, model_name: str = "deepseek/deepseek-chat", max_depth: int = 5):
        self.model_name = model_name
        self.max_depth = max_depth
        self.logger = logging.getLogger(__name__)

        # Graph traversal directions and edge types
        self.edge_types = [EDGE_TYPE_CONTAINS, EDGE_TYPE_IMPORTS, EDGE_TYPE_INVOKES, EDGE_TYPE_INHERITS]
        self.edge_directions = ['downstream', 'upstream']

        # Initialize the positioning chain embedding calculator
        self.chain_embedding = LocalizationChainEmbedding()
        
        # Cache related configuration
        self.cache_dir = "/entity_pipeline_cache"
        self.enable_cache = True
        self._ensure_cache_dir_exists()
        
        # Initialize the OpenAI client
        self.client = OpenAI(
            base_url="",
            api_key="",
            timeout=60.0
        )

    def _call_llm_simple(self, messages: List[Dict], temp: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Simple LLM call function, reusing the logic in entity_search_scorer.py

        Args:
            messages: Message list
            temp: Temperature parameter
            max_tokens: Maximum token count

        Returns:
            Model response text
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-v3",
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                timeout=60
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise e

    def run_pipeline(self, instance_data: Dict[str, Any], context: str, max_initial_entities: int = 5) -> str:
        """
        Run the complete entity localization pipeline.

        Args:
            instance_data: Instance data containing problem statement and metadata
            context: Context string containing initial entities
            max_initial_entities: Maximum number of initial entities to extract

        Returns:
            Dictionary containing grouped localization chains, final voted result, modification plan, and formatted edit prompt
        """
        logging.info("=" * 80)
        logging.info("=== Starting Entity Localization Pipeline ===")
        logging.info(f"Instance ID: {instance_data.get('instance_id', 'unknown')}")
        logging.info(f"Repository: {instance_data.get('repo', 'unknown')}")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Max depth: {self.max_depth}")
        logging.info(f"Max initial entities: {max_initial_entities}")
        logging.info(f"Context length: {len(context)}")
        logging.info(f"Problem statement length: {len(instance_data.get('problem_statement', ''))}")
        logging.info("=" * 80)

        # Setup current issue
        set_current_issue(instance_data=instance_data)
        logging.info("Current issue setup completed")

        # Store issue descriptions for use by LLM
        self._current_issue_description = instance_data['problem_statement']
        
        # Generate cache timestamp
        cache_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        instance_id = instance_data.get('instance_id', 'unknown')
        
        logging.info(f"cache timestamp: {cache_timestamp}")

        try:
            # Stage 1: Extract initial entities from context
            logging.info("Starting Stage 1: Extracting initial entities from context")
            initial_entities = self._extract_initial_entities(context, instance_data['problem_statement'],
                                                              max_initial_entities)
            logging.info(f"Stage 1 completed, extracted {len(initial_entities)} initial entities: {initial_entities}")

            # Save Stage 1 results
            stage1_data = {
                'initial_entities': initial_entities,
                'context': context,
                'max_initial_entities': max_initial_entities,
                'context_length': len(context),
                'problem_statement_length': len(instance_data.get('problem_statement', ''))
            }
            self._save_stage_result(instance_id, 'stage_1_initial_entities', stage1_data, cache_timestamp)

            if not initial_entities:
                logging.warning("No initial entities were extracted, the process ends")
                return {
                    'instance_id': instance_data['instance_id'],
                    'context': context,
                    'initial_entities': [],
                    'grouped_localization_chains': [],
                    'selected_chains': [],
                    'total_chains': 0,
                    'error': 'No initial entities extracted',
                    'metadata': {
                        'repo': instance_data['repo'],
                        'base_commit': instance_data['base_commit'],
                        'problem_statement': instance_data['problem_statement']
                    }
                }

            # Stage 2: For each initial entity, search code snippets and extract related entities
            logging.info("Start phase 2: Search for related entities for each initial entity")
            entity_groups = []
            for i, initial_entity in enumerate(initial_entities):
                logging.info(f"Processing initial entity {i + 1}/{len(initial_entities)}: '{initial_entity}'")
                related_entities = self._extract_related_entities_for_initial_entity(
                    initial_entity, instance_data['problem_statement']
                )
                entity_groups.append({
                    'initial_entity': initial_entity,
                    'related_entities': related_entities
                })
                logging.info(f"for '{initial_entity}' find {len(related_entities)} related entities")

            total_related_entities = sum(len(group['related_entities']) for group in entity_groups)
            logging.info(f"Phase 2 completed, a total of {total_related_entities} related entities found")

            # Save Stage 2 results
            stage2_data = {
                'entity_groups': entity_groups,
                'total_related_entities': total_related_entities
            }
            self._save_stage_result(instance_id, 'stage_2_related_entities', stage2_data, cache_timestamp)

            # Stage 3: Generate localization chains for each related entity, grouped by initial entity 
            logging.info("Start phase 3: Generate location chains for related entities (parallel processing)")
            grouped_localization_chains = []
            all_chains = []  # Collect all positioning chains for subsequent selection
            total_chains_generated = 0
            results_lock = threading.Lock()

            def process_group_worker(group_index: int, group: Dict[str, Any]) -> Dict[str, Any]:
                """A single group's positioning chain generates a working function"""
                initial_entity = group['initial_entity']
                logging.info(f"Start processing group {group_index + 1}/{len(entity_groups)} - Initial entity: '{initial_entity}'")

                try:
                    localization_chains = self._generate_localization_chains(group['related_entities'])
                    
                    group_result = {
                        'initial_entity': initial_entity,
                        'related_entities': group['related_entities'],
                        'localization_chains': localization_chains,
                        'chain_count': len(localization_chains),
                        'group_index': group_index
                    }

                    # Collect valid positioning chains for this group
                    group_valid_chains = []
                    for chain_info in localization_chains:
                        if chain_info.get('chain') and len(chain_info['chain']) > 0:
                            group_valid_chains.append({
                                'chain': chain_info['chain'],
                                'chain_length': chain_info['chain_length'],
                                'initial_entity': initial_entity,
                                'start_entity': chain_info['start_entity']
                            })
                    
                    group_result['valid_chains'] = group_valid_chains

                    logging.info(f"Group {group_index + 1} ('{initial_entity}') processing completed, generated {len(localization_chains)} positioning chains")
                    return group_result
                    
                except Exception as e:
                    logging.error(f"Group {group_index + 1} ('{initial_entity}') processing failed: {e}")
                    return {
                        'initial_entity': initial_entity,
                        'related_entities': group['related_entities'],
                        'localization_chains': [],
                        'chain_count': 0,
                        'group_index': group_index,
                        'valid_chains': [],
                        'error': str(e)
                    }

            # Use a thread pool to process all groups in parallel
            max_workers = min(len(entity_groups), 5)  # Limit the number of concurrent connections to avoid excessive resource usage
            logging.info(f"Start {max_workers} worker threads for parallel group processing")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_group_worker, i, group)
                    for i, group in enumerate(entity_groups)
                ]

                completed_results = []
                for future in as_completed(futures):
                    try:
                        group_result = future.result()
                        with results_lock:
                            completed_results.append(group_result)
                    except Exception as e:
                        logging.error(f"Group processing thread exception: {e}")

            completed_results.sort(key=lambda x: x['group_index'])
            
            for group_result in completed_results:
                final_group_result = {
                    'initial_entity': group_result['initial_entity'],
                    'related_entities': group_result['related_entities'],
                    'localization_chains': group_result['localization_chains'],
                    'chain_count': group_result['chain_count']
                }
                grouped_localization_chains.append(final_group_result)
                
                all_chains.extend(group_result.get('valid_chains', []))
                total_chains_generated += group_result['chain_count']

            for i, chain in enumerate(all_chains, 1):
                logging.info("Chain %d: %s", i, chain)
            
            stage3_data = {
                'grouped_localization_chains': grouped_localization_chains,
                'all_chains': all_chains,
                'total_chains_generated': total_chains_generated
            }
            self._save_stage_result(instance_id, 'stage_3_localization_chains', stage3_data, cache_timestamp)

            # Stage 4: Use embedding to select diverse localization chains
            selected_chains = self._select_diverse_chains(all_chains)

            # Save Stage 4 results
            stage4_data = {
                'selected_chains': selected_chains,
                'selected_chains_count': len(selected_chains)
            }
            self._save_stage_result(instance_id, 'stage_4_diverse_chains', stage4_data, cache_timestamp)

            # Stage 5: Add code information to the selected positioning chain
            chains_with_code = self._add_code_to_chains(selected_chains)
            
            stage5_data = {
                'chains_with_code': chains_with_code,
                'chains_with_code_count': len(chains_with_code)
            }
            self._save_stage_result(instance_id, 'stage_5_chains_with_code', stage5_data, cache_timestamp)

            # Stage 6: Use multiple agents to vote on the localization chains
            voting_result = self._vote_on_chains(chains_with_code, instance_data['problem_statement'])

            # Save Stage 6 results (including winning chain information)
            stage6_data = {
                'voting_result': voting_result,
                'chains_with_code': chains_with_code,  
                'winning_chain': voting_result.get('winning_chain'),
                'winning_chain_id': voting_result.get('winning_chain_id')
            }
            self._save_stage_result(instance_id, 'stage_6_voting_result', stage6_data, cache_timestamp)

            if not voting_result.get('success') or not voting_result.get('winning_chain'):
                return self._create_error_result(instance_data, context, 'No winning chain found')

            # Stage 7: Use multi-turn agent discussion to generate modification plan
            modification_plan = self._generate_modification_plan(
                voting_result['winning_chain'],
                instance_data['problem_statement'],
                5,  # num_agents
                instance_id,
                cache_timestamp
            )

            # Stage 8: Format output for edit agent
            edit_agent_prompt = self._format_edit_agent_prompt(
                instance_data['problem_statement'],
                modification_plan,
                voting_result['winning_chain']
            )
            
            stage8_data = {
                'edit_agent_prompt': edit_agent_prompt,
                'modification_plan': modification_plan,
                'winning_chain': voting_result['winning_chain']
            }
            self._save_stage_result(instance_id, 'stage_8_edit_agent_prompt', stage8_data, cache_timestamp)

            result = {
                'instance_id': instance_data['instance_id'],
                'context': context,
                'initial_entities': initial_entities,
                'grouped_localization_chains': grouped_localization_chains,
                'selected_chains': selected_chains,
                'chains_with_code': chains_with_code,
                'voting_result': voting_result,
                'final_selected_chain': voting_result.get('winning_chain'),
                'modification_plan': modification_plan,
                'edit_agent_prompt': edit_agent_prompt,  
                'total_chains': len(all_chains),
                'metadata': {
                    'repo': instance_data['repo'],
                    'base_commit': instance_data['base_commit'],
                    'problem_statement': instance_data['problem_statement']
                }
            }

            logging.info("=" * 80)
            logging.info("=== Entity Localization Pipeline completed ===")
            logging.info("=" * 80)

            # return result
            return edit_agent_prompt

        finally:
            # Cleanup
            reset_current_issue()
            logging.info("Current issue cleanup completed")

    def _create_error_result(self, instance_data: Dict[str, Any], context: str, error_msg: str) -> Dict[str, Any]:
        return {
            'instance_id': instance_data['instance_id'],
            'context': context,
            'initial_entities': [],
            'grouped_localization_chains': [],
            'selected_chains': [],
            'total_chains': 0,
            'error': error_msg,
            'metadata': {
                'repo': instance_data['repo'],
                'base_commit': instance_data['base_commit'],
                'problem_statement': instance_data['problem_statement']
            }
        }

    def _extract_initial_entities(self, context: str, issue_description: str, max_entities: int = 5) -> List[str]:
        """
        Stage 1: Extract initial entities from context using ENTITY_EXTRACTION_PROMPT.

        Args:
            context: Context string containing potential entities
            issue_description: Problem statement from issue
            max_entities: Maximum number of entities to extract

        Returns:
            List of initial entity names (limited to max_entities)
        """
        logging.info("=== Stage 1: Extracting Initial Entities from Context ===")

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            context=context,
            issue_description=issue_description,
            max_entities=max_entities
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self._call_llm_simple(
                messages=messages,
                temp=0.7,
                max_tokens=1000
            )

            # Parsing JSON Response
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                logging.info("Removed opening ```json tag")
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                logging.info("Removed closing ``` tag")

            logging.info(f"Cleaned response: {response_text}")
            entities = json.loads(response_text)

            if isinstance(entities, list) and all(isinstance(entity, str) for entity in entities):
                entities = entities[:max_entities]
                return entities
            else:
                logging.warning(f"The large model return format is incorrect, the return type: {type(entities)}")
                return []

        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}, original response: {response}")
            return []
        except Exception as e:
            logging.error(f"Initial entity extraction failed: {e}")
            return []

    def _extract_related_entities_for_initial_entity(self, initial_entity: str, problem_statement: str) -> List[
        Dict[str, Any]]:
        """
        Stage 2: For each initial entity, search code snippets and extract 4 related entities.

        Args:
            initial_entity: Initial entity name to search for
            problem_statement: Problem statement from issue

        Returns:
            List of 4 related entities with their metadata
        """
        logging.info(f"=== Stage 2: Extracting Related Entities for '{initial_entity}' ===")

        # Search for code snippets containing this entity
        code_snippets = self._search_code_snippets_for_entity(initial_entity)

        if not code_snippets or len(code_snippets.strip()) == 0:
            logging.warning(f"No code snippets found for entity '{initial_entity}'")
            return []

        # Extract 4 related entities from the code snippets
        logging.info(f"Extracting related entities from code snippets...")
        related_entities = self._extract_entities_from_code_snippets(code_snippets, problem_statement)

        logging.info(f"Found {len(related_entities)} related entities for '{initial_entity}'")
        for i, entity in enumerate(related_entities):
            logging.info(
                f"entity {i + 1}: {entity.get('entity_id', 'unknown')} ({entity.get('entity_type', 'unknown')})"
            )
            logging.info(f"reason: {entity.get('relevance_reason', 'No reason provided')}")

        return related_entities

    def _search_code_snippets_for_entity(self, entity_name: str) -> str:
        """
        Search for code snippets containing the given entity name.

        Args:
            entity_name: Entity name to search for

        Returns:
            Found code snippets as string
        """
        # Use the entity name as search term
        search_terms = [entity_name]

        # Also try common variations
        variations = [
            entity_name,
            f"def {entity_name}",
            f"class {entity_name}",
            f".{entity_name}(",
            f"_{entity_name}",
        ]
        search_terms.extend(variations)

        logging.info(f"Search word list: {search_terms}")

        # Search for code snippets
        logging.info("Call search_code_snippets to search...")
        code_snippets = search_code_snippets(
            search_terms=search_terms,
            file_path_or_pattern="**/*.py"
        )

        logging.info(f"Search completed, find the length of the code snippet: {len(code_snippets)}")
        if code_snippets:
            logging.info(f"Code snippet preview: {code_snippets[:200]}...")

        return code_snippets

    def _extract_entities_from_code_snippets(self, code_snippets: str, problem_statement: str) -> List[Dict[str, Any]]:
        """
        Extract 4 most relevant entities from code snippets using CODE_SNIPPET_ENTITY_EXTRACTION_PROMPT.

        Args:
            code_snippets: Found code snippets
            problem_statement: Problem statement from issue

        Returns:
            List of extracted entities with their metadata
        """
        logging.info("Start extracting entities from code snippets...")

        prompt = CODE_SNIPPET_ENTITY_EXTRACTION_PROMPT.format(
            code_snippets=code_snippets,
            problem_statement=problem_statement
        )

        logging.info(f"The length of the constructed entity extraction prompt: {len(prompt)}")

        messages = [
            {
                "role": "system",
                "content": "You are an expert code analysis assistant that can identify the most relevant entities for solving software issues."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            logging.info("Call LLM to extract entity...")
            response = self.client.chat.completions.create(
                model="deepseek-v3",
                messages=messages,
                temperature=0.7,
            )

            entities_text = response.choices[0].message.content

            entities = self._parse_extracted_entities(entities_text)

            logging.info(f"Successfully extracted {len(entities)} entities from code snippets")
            return entities

        except Exception as e:
            logging.error(f"Entity extraction from code snippets failed: {e}")
            return []

    def _parse_extracted_entities(self, entities_text: str) -> List[Dict[str, Any]]:
        """Parse extracted entities from LLM response."""
        logging.info("Start parsing the entities returned by LLM...")
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*?\]', entities_text, re.DOTALL)
            if json_match:
                entities_json = json_match.group()
                entities = json.loads(entities_json)

                # Validate and clean entities
                validated_entities = []
                for i, entity in enumerate(entities[:4]):  # Ensure max 4 entities
                    if 'entity_id' in entity and 'entity_type' in entity:
                        validated_entities.append(entity)
                        logging.info(f"Verify entity {i + 1}: {entity['entity_id']} ({entity['entity_type']})")
                    else:
                        logging.warning(f"Skip invalid entity {i + 1}: {entity}")

                logging.info(f"Validation complete, valid entities: {len(validated_entities)}")
                return validated_entities
            else:
                logging.warning("No JSON format data found in response")
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}")
        except Exception as e:
            logging.error(f"Failed to parse entities: {e}")

        return []

    def _select_next_node_with_llm(self, current_entity: str, neighbors: List[str],
                                   issue_description: str, entity_searcher, depth: int) -> Dict[str, Any]:
        """
        Use LLM to select the next most promising node for exploration

        Args:
            current_entity: Current entity ID
            neighbors: List of neighbor node IDs
            issue_description: Issue description
            entity_searcher: Entity searcher
            depth: Current depth

        Returns:
            A dictionary containing the selection decision
        """
        logging.info(f"Use LLM to select the next node, current entity: {current_entity}, depth: {depth}, number of neighbors: {len(neighbors)}")

        if not neighbors:
            logging.info("No available neighbors, stopping exploration")
            return {"should_continue": False, "selected_neighbor": None, "reasoning": "No neighbors available"}

        try:
            current_entity_data = entity_searcher.get_node_data([current_entity])[0]
            current_entity_type = current_entity_data['type']

            logging.info(f"Current entity type: {current_entity_type}")

            neighbor_info_list = []
            for i, neighbor in enumerate(neighbors[:10]):  
                try:
                    neighbor_data = entity_searcher.get_node_data([neighbor])[0]
                    neighbor_info = f"- {neighbor} (Type: {neighbor_data['type']})"
                    if 'code' in neighbor_data and neighbor_data['code']:
                        code_preview = neighbor_data['code'][:500] + "..." if len(neighbor_data['code']) > 500 else \
                            neighbor_data['code']
                        neighbor_info += f"\n  Code preview: {code_preview}"
                    neighbor_info_list.append(neighbor_info)
                    logging.info(f" {i + 1}: {neighbor} ({neighbor_data['type']})")
                except:
                    neighbor_info_list.append(f"- {neighbor} (Type: unknown)")
                    logging.warning(f"Neighbor {i + 1}: {neighbor} (failed to get information)")

            neighbor_info = "\n".join(neighbor_info_list)

            # Build prompt
            prompt = NODE_SELECTION_PROMPT.format(
                issue_description=issue_description,
                current_entity=current_entity,
                current_entity_type=current_entity_type,
                depth=depth,
                neighbor_info=neighbor_info
            )

            logging.info(f"The length of the constructed node selection prompt: {len(prompt)}")
            messages = [{"role": "user", "content": prompt}]

            logging.info("Calling LLM for node selection...")
            response = self._call_llm_simple(
                messages=messages,
                temp=0.7,
                max_tokens=1000
            )

            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            decision = json.loads(response_text)

            if isinstance(decision, dict) and 'should_continue' in decision:
                should_continue = decision.get('should_continue', False)
                selected_neighbor = decision.get('selected_neighbor')
                reasoning = decision.get('reasoning', 'No reasoning provided')
                logging.info(f"LLM decision: continue exploration={should_continue}, selected neighbor={selected_neighbor}")
                logging.info(f"LLM reasoning: {reasoning}")
                return decision
            else:
                logging.warning("LLM response format is incorrect, using fallback logic")
                return self._fallback_node_selection(current_entity, neighbors, entity_searcher, depth)

        except Exception as e:
            logging.error(f"LLM node selection failed: {e}, using fallback logic")
            return self._fallback_node_selection(current_entity, neighbors, entity_searcher, depth)

    def _prefilter_neighbors_with_llm(self, current_entity: str, all_neighbors: List[Dict],
                                      issue_description: str, entity_searcher, depth: int,
                                      max_selection: int = 10) -> List[str]:
        """
        Use LLM to pre-screen neighbors and select the most relevant and diverse neighbor entities

        Args:
            current_entity: Current entity ID
            all_neighbors: List of all neighbor entity information
            issue_description: Issue description
            entity_searcher: Entity searcher
            depth: Current depth
            max_selection: Maximum selection count

        Returns:
            List of filtered neighbor entity IDs
        """
        if len(all_neighbors) <= max_selection:
            return [n['entity_id'] for n in all_neighbors]

        logging.info(f"Start pre-screening neighbors, total: {len(all_neighbors)}, target selection: {max_selection}")

        try:
            current_entity_data = entity_searcher.get_node_data([current_entity])[0]
            current_entity_type = current_entity_data['type']

            neighbor_list_parts = []
            for i, neighbor_info in enumerate(all_neighbors):
                neighbor_id = neighbor_info['entity_id']
                edge_type = neighbor_info['edge_type']
                direction = neighbor_info['direction']

                try:
                    neighbor_data = entity_searcher.get_node_data([neighbor_id])[0]
                    neighbor_type = neighbor_data['type']
                    neighbor_list_parts.append(
                        f"{i + 1}. {neighbor_id} (Type: {neighbor_type}, Edge: {edge_type}, Direction: {direction})"
                    )
                except:
                    neighbor_list_parts.append(
                        f"{i + 1}. {neighbor_id} (Type: unknown, Edge: {edge_type}, Direction: {direction})"
                    )

            neighbor_list = "\n".join(neighbor_list_parts)

            prompt = NEIGHBOR_PREFILTERING_PROMPT.format(
                issue_description=issue_description,
                current_entity=current_entity,
                current_entity_type=current_entity_type,
                depth=depth,
                total_count=len(all_neighbors),
                neighbor_list=neighbor_list,
                max_selection=max_selection
            )

            logging.info(f"The length of the neighbor pre-screening prompt constructed: {len(prompt)}")
            messages = [{"role": "user", "content": prompt}]

            response = self._call_llm_simple(
                messages=messages,
                temp=0.7,
                max_tokens=1000
            )

            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            selection_result = json.loads(response_text)

            if isinstance(selection_result, dict) and 'selected_neighbors' in selection_result:
                selected_neighbors = selection_result['selected_neighbors']
                reasoning = selection_result.get('selection_reasoning', 'No reasoning provided')
                diversity = selection_result.get('diversity_considerations', 'No diversity info')

                available_neighbor_ids = [n['entity_id'] for n in all_neighbors]
                valid_selected = [n for n in selected_neighbors if n in available_neighbor_ids]

                logging.info(f"LLM selected {len(valid_selected)} neighbors")
                logging.info(f"Selection reasoning: {reasoning}")
                logging.info(f"Diversity considerations: {diversity}")
                logging.info(f"Selected neighbors: {valid_selected}")

                return valid_selected
            else:
                logging.warning("LLM pre-screening returned invalid format, using fallback strategy")
                return self._fallback_neighbor_prefiltering(all_neighbors, max_selection)

        except Exception as e:
            logging.error(f"LLM neighbor pre-screening failed: {e}, using fallback strategy")
            return self._fallback_neighbor_prefiltering(all_neighbors, max_selection)

    def _fallback_neighbor_prefiltering(self, all_neighbors: List[Dict], max_selection: int) -> List[str]:
        """
        Fallback neighbor pre-screening strategy

        Args:
            all_neighbors: List of all neighbor entity information
            max_selection: Maximum number of selections

        Returns:
            Filtered neighbor entity ID list
        """
        logging.info("Using fallback neighbor pre-screening strategy")

        #logging.info("Use fallback neighbor pre-screening strategy")
        selected = []
        seen_files = set()

        type_priority = {'function': 0, 'class': 1, 'file': 2}

        for neighbor_info in all_neighbors:
            if len(selected) >= max_selection:
                break

            neighbor_id = neighbor_info['entity_id']
            file_path = neighbor_id.split(':')[0] if ':' in neighbor_id else neighbor_id

            if file_path not in seen_files:
                selected.append(neighbor_id)
                seen_files.add(file_path)

        for neighbor_info in all_neighbors:
            if len(selected) >= max_selection:
                break

            neighbor_id = neighbor_info['entity_id']
            if neighbor_id not in selected:
                selected.append(neighbor_id)

        return selected

    def _dfs_traversal(self, start_entity: str, graph, entity_searcher, dependency_searcher,
                       max_depth: int) -> List[str]:
        """
        Perform DFS traversal starting from an entity to find the best localization chain.

        Args:
            start_entity: Starting entity ID
            graph: Repository graph
            entity_searcher: Entity searcher instance
            dependency_searcher: Dependency searcher instance
            max_depth: Maximum traversal depth

        Returns:
            Simplified localization chain as list of entity IDs only
        """
        logging.info(f"Start DFS traversal, starting entity: {start_entity}, maximum depth: {max_depth}")

        visited = set()
        best_chain = []
        issue_description = getattr(self, '_current_issue_description', '')

        def dfs(current_entity: str, depth: int, current_path: List[str]) -> bool:
            """
            Recursive DFS function with LLM-guided node selection.

            Returns:
                True if we should stop traversal (found target or reached limit)
            """
            nonlocal best_chain

            logging.info(f"DFS visiting node: {current_entity}, depth: {depth}, path length: {len(current_path)}")

            if depth >= max_depth or current_entity in visited:
                if depth >= max_depth:
                    logging.info(f"Reached maximum depth {max_depth}, stopping exploration")
                if current_entity in visited:
                    logging.info(f"Node {current_entity} has been visited, skipping")
                return False

            visited.add(current_entity)

            # Add current step to path - only save entity_id
            current_path.append(current_entity)

            all_neighbors = []
            for direction in self.edge_directions:
                for edge_type in self.edge_types:
                    try:
                        if direction == 'downstream':
                            neighbors, edges = dependency_searcher.get_neighbors(
                                current_entity, 'forward', etype_filter=[edge_type]
                            )
                        else:  # upstream
                            neighbors, edges = dependency_searcher.get_neighbors(
                                current_entity, 'backward', etype_filter=[edge_type]
                            )

                        for neighbor in neighbors:
                            if neighbor not in visited:
                                all_neighbors.append({
                                    'entity_id': neighbor,
                                    'edge_type': edge_type,
                                    'direction': direction
                                })

                    except Exception as e:
                        logging.debug(f"Error exploring {direction} {edge_type} from {current_entity}: {e}")
                        continue

            if not all_neighbors:
                if len(current_path) > len(best_chain):
                    best_chain = current_path.copy()
                return True

            if len(all_neighbors) > 10:
                prefiltered_neighbors = self._prefilter_neighbors_with_llm(
                    current_entity, all_neighbors, issue_description, entity_searcher, depth, max_selection=10
                )
            else:
                prefiltered_neighbors = [n['entity_id'] for n in all_neighbors]


            decision = self._select_next_node_with_llm(
                current_entity, prefiltered_neighbors, issue_description, entity_searcher, depth
            )

            if not decision.get('should_continue', False):
                if len(current_path) > len(best_chain):
                    best_chain = current_path.copy()
                return True

            selected_neighbor = decision.get('selected_neighbor')
            if selected_neighbor and selected_neighbor in prefiltered_neighbors:
                if dfs(selected_neighbor, depth + 1, current_path.copy()):
                    return True

            backup_count = 0
            for neighbor in prefiltered_neighbors[:3]:
                if neighbor != selected_neighbor and neighbor not in visited:
                    backup_count += 1

                    if dfs(neighbor, depth + 1, current_path.copy()):
                        return True

            return False

        issue_description = self._current_issue_description

        # Start DFS
        dfs(start_entity, 0, [])

        return best_chain

    def _generate_localization_chains(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 3: Generate localization chains for each entity using DFS graph traversal (parallel version).

        Args:
            entities: List of extracted entities

        Returns:
            List of localization chains, one for each entity
        """
        logging.info("=== Stage 3: Generating Localization Chains (Parallel) ===")

        if not entities:
            logging.warning("No entities are used to generate positioning chains")
            return []

        graph = get_graph()
        entity_searcher = get_graph_entity_searcher()
        dependency_searcher = get_graph_dependency_searcher()


        all_chains = []
        chain_lock = threading.Lock()

        def generate_chain_worker(entity_index: int, entity: Dict[str, Any]) -> Dict[str, Any]:
            """
            Generate worksheet for location chain of a single entity

            Args:
                entity_index: The index of the entity in the list
                entity: Entity information dictionary

            Returns:
                A dictionary containing information about the location chain
            """
            entity_id = entity['entity_id']
            logging.info(f"entity {entity_index + 1}/{len(entities)}: {entity_id}")

            try:
                if not entity_searcher.has_node(entity_id):
                    logging.warning(f"Entity {entity_id} not found in graph")
                    return {
                        'start_entity': entity,
                        'chain': [],
                        'chain_length': 0,
                        'error': 'Entity not found in graph'
                    }

                # Perform DFS traversal for this entity (now returns simplified chain)
                chain = self._dfs_traversal(
                    start_entity=entity_id,
                    graph=graph,
                    entity_searcher=entity_searcher,
                    dependency_searcher=dependency_searcher,
                    max_depth=self.max_depth
                )

                chain_info = {
                    'start_entity': entity,
                    'chain': chain,  # Now contains only entity IDs
                    'chain_length': len(chain)
                }

                if chain:
                    logging.info(f"Simplified positioning chain: {chain}")

                return chain_info

            except Exception as e:
                logging.error(f"Entity {entity_id} Location chain generation failed: {e}")
                return {
                    'start_entity': entity,
                    'chain': [],
                    'chain_length': 0,
                    'error': f'Chain generation failed: {e}'
                }

        max_workers = min(len(entities), 5)  

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(generate_chain_worker, i, entity)
                for i, entity in enumerate(entities)
            ]

            for future in as_completed(futures):
                try:
                    chain_info = future.result()
                    with chain_lock:
                        all_chains.append(chain_info)
                except Exception as e:
                    logging.error(f"Positioning chain generates thread exception: {e}")
                    with chain_lock:
                        all_chains.append({
                            'start_entity': {'entity_id': 'unknown'},
                            'chain': [],
                            'chain_length': 0,
                            'error': f'Thread execution failed: {e}'
                        })

        entity_ids_order = [entity['entity_id'] for entity in entities]
        ordered_chains = []

        for entity_id in entity_ids_order:
            matching_chain = None
            for chain_info in all_chains:
                if chain_info['start_entity'].get('entity_id') == entity_id:
                    matching_chain = chain_info
                    break

            if matching_chain:
                ordered_chains.append(matching_chain)
            else:
                logging.warning(f"No location chain result found for entity {entity_id}")
                ordered_chains.append({
                    'start_entity': {'entity_id': entity_id},
                    'chain': [],
                    'chain_length': 0,
                    'error': 'Result not found after parallel execution'
                })

        logging.info(f"All positioning chains have been generated, total: {len(ordered_chains)}")

        successful_chains = [c for c in ordered_chains if c.get('chain') and len(c['chain']) > 0]
        failed_chains = [c for c in ordered_chains if c.get('error')]

        logging.info(f"successful_chains: {len(successful_chains)}")
        logging.info(f"failed_chains: {len(failed_chains)}")

        return ordered_chains

    def _select_diverse_chains(self, all_chains: List[Dict[str, Any]], max_selected: int = 6) -> List[Dict[str, Any]]:
        """
        Stage 4: Use embedding to select diverse positioning chains (based on the longest chain)

        Args:
            all_chains: All generated positioning chains
            max_selected: Maximum selection quantity (including the longest chain)

        Returns:
            List of selected positioning chains
        """
        logging.info(f"=== Stage 4: Use embedding to select diverse positioning chains (based on the longest chain) ===")

        if not all_chains:
            logging.warning("No locating chain available")
            return []

        non_empty_chains = [chain for chain in all_chains if chain.get('chain') and len(chain['chain']) > 0]
        logging.info(f"Number of non-empty positioning chains: {len(non_empty_chains)}")

        if len(non_empty_chains) <= max_selected:
            logging.info(f"Number of non-empty positioning chains({len(non_empty_chains)}) not exceed the maximum number of selections({max_selected}) return all")
            return non_empty_chains

        chains_for_embedding = [chain_info['chain'] for chain_info in all_chains]

        try:
            selected_indices, similarity_scores = self.chain_embedding.select_diverse_chains(
                chains_for_embedding, k=max_selected - 1  
            )

            if not selected_indices:
                sorted_chains = sorted(non_empty_chains, key=lambda x: x['chain_length'], reverse=True)
                return sorted_chains[:max_selected]

            selected_chains = []
            for i, idx in enumerate(selected_indices):
                if idx >= len(all_chains):
                    logging.warning(f"Index {idx} out of range, skipping")
                    continue

                chain_info = deepcopy(all_chains[idx])
                chain_info['selection_rank'] = i + 1
                chain_info['similarity_to_longest'] = similarity_scores[i] if i < len(similarity_scores) else 1.0
                chain_info['is_longest'] = (i == 0)  
                selected_chains.append(chain_info)

            for i, chain in enumerate(selected_chains):
                logging.info(f"content: {chain['chain']}")

            return selected_chains

        except Exception as e:
            logging.error(f"Failed to select the positioning chain: {e}, return the chain sorted by length")
            sorted_chains = sorted(non_empty_chains, key=lambda x: x['chain_length'], reverse=True)
            return sorted_chains[:max_selected]

    def _fallback_node_selection(self, current_entity: str, neighbors: List[str], entity_searcher, depth: int) -> Dict[
        str, Any]:
        """
        Fallback logic for LLM node selection

        Args:
            current_entity: Current entity ID
            neighbors: List of neighboring nodes
            entity_searcher: Entity searcher
            depth: Current depth

        Returns:
            A dictionary containing the selection decisions
        """

        if not neighbors:
            return {"should_continue": False, "selected_neighbor": None, "reasoning": "No neighbors available"}

        if depth >= self.max_depth - 1:
            return {
                "should_continue": False,
                "selected_neighbor": None,
                "reasoning": f"Reached maximum depth {self.max_depth}"
            }

        selected_neighbor = neighbors[0]

        return {
            "should_continue": True,
            "selected_neighbor": selected_neighbor,
            "reasoning": f"Fallback selection: chose first neighbor {selected_neighbor}"
        }

    def _add_code_to_chains(self, selected_chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 5: Add code information to the selected positioning chain

        Args:
            selected_chains: List of selected positioning chains

        Returns:
            A list of location chains containing code information
        """
        logging.info(f"=== Stage 5: Add code information for {len(selected_chains)} location chains ===")

        chains_with_code = []
        entity_searcher = get_graph_entity_searcher()
        MAX_CODE_LENGTH = 10000 

        for i, chain_info in enumerate(selected_chains):
            chain_id = f"chain_{i + 1}"
            chain = chain_info['chain']

            logging.info(f"Processing chain {chain_id}, length: {len(chain)}")

            entities_with_code = []
            for j, entity_id in enumerate(chain):
                logging.info(f"Get entity {j + 1}/{len(chain)}: {entity_id}")

                try:
                    if entity_searcher.has_node(entity_id):
                        entity_data = entity_searcher.get_node_data([entity_id], return_code_content=True)[0]
                        code_content = entity_data.get('code_content', '')

                        if len(code_content) > MAX_CODE_LENGTH:
                            logging.info(
                                f"Entity code length {len(code_content)} exceeds the limit {MAX_CODE_LENGTH}, only the entity name is retained")
                            entity_with_code = {
                                'entity_id': entity_id,
                                'type': entity_data['type'],
                                'code': f'# Code too long ({len(code_content)} chars) - omitted for brevity',
                                'start_line': entity_data.get('start_line'),
                                'end_line': entity_data.get('end_line'),
                                'code_omitted': True,
                                'original_code_length': len(code_content)
                            }
                        else:
                            entity_with_code = {
                                'entity_id': entity_id,
                                'type': entity_data['type'],
                                'code': code_content,
                                'start_line': entity_data.get('start_line'),
                                'end_line': entity_data.get('end_line'),
                                'code_omitted': False,
                                'original_code_length': len(code_content)
                            }

                        entities_with_code.append(entity_with_code)
                        logging.info(
                            f" Successfully obtained code, length: {len(entity_with_code['code'])} (original length: {len(code_content)})")
                    else:
                        logging.warning(f"Entity {entity_id} does not exist in the graph")
                        entities_with_code.append({
                            'entity_id': entity_id,
                            'type': 'unknown',
                            'code': '# Entity not found in graph',
                            'start_line': None,
                            'end_line': None,
                            'code_omitted': False,
                            'original_code_length': 0
                        })
                except Exception as e:
                    logging.error(f"Get entity {entity_id} code failed: {e}")
                    entities_with_code.append({
                        'entity_id': entity_id,
                        'type': 'error',
                        'code': f'# Error getting code: {e}',
                        'start_line': None,
                        'end_line': None,
                        'code_omitted': False,
                        'original_code_length': 0
                    })

            chain_with_code = {
                'chain_id': chain_id,
                'original_chain_info': chain_info,
                'entities_with_code': entities_with_code,
                'chain_length': len(entities_with_code),
                'selection_rank': chain_info.get('selection_rank', i + 1),
                'is_longest': chain_info.get('is_longest', False)
            }

            chains_with_code.append(chain_with_code)

            omitted_count = sum(1 for entity in entities_with_code if entity.get('code_omitted', False))
            total_code_length = sum(entity.get('original_code_length', 0) for entity in entities_with_code)

            logging.info(f"Chain {chain_id} processing complete, containing {len(entities_with_code)} entity codes")
            logging.info(f"  Among them, {omitted_count} entity codes were omitted, total code length: {total_code_length}")

        logging.info(f"Stage 5 complete, all {len(chains_with_code)} chains have added code information")
        return chains_with_code

    def _vote_on_chains(self, chains_with_code: List[Dict[str, Any]], issue_description: str, num_agents: int = 5) -> \
            Dict[str, Any]:
        """
        Stage 6: Use multiple agents to vote on the positioning chain

        Args:
            chains_with_code: List of location chains containing code
            issue_description: Problem Description
            num_agents: Number of voting agents

        Returns:
            vote results
        """
        logging.info(f"=== Stage 6: Use {num_agents} agents to vote on {len(chains_with_code)} location chains ===")

        if not chains_with_code:
            return {
                'success': False,
                'error': 'No chains to vote on',
                'votes': [],
                'winning_chain': None
            }

        chains_info = self._format_chains_for_voting(chains_with_code)

        vote_results = []
        vote_lock = threading.Lock()

        def vote_worker(agent_id: int) -> Dict[str, Any]:
            try:
                logging.info(f"Agent {agent_id} begin...")

                prompt = CHAIN_VOTING_PROMPT.format(
                    issue_description=issue_description,
                    chains_info=chains_info
                )

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert software engineer with deep experience in code analysis and debugging."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                response = self._call_llm_simple(
                    messages=messages,
                    temp=0.7,  
                    max_tokens=2000
                )

                vote_result = self._parse_vote_result(response, agent_id)

                with vote_lock:
                    vote_results.append(vote_result)
                    logging.info(f"Agent {agent_id} complete: {vote_result.get('voted_chain_id', 'unknown')}")

                return vote_result

            except Exception as e:
                logging.error(f"Agent {agent_id} fail: {e}")
                return {
                    'agent_id': agent_id,
                    'voted_chain_id': None,
                    'confidence': 0,
                    'reasoning': f'fail: {e}',
                    'error': str(e)
                }

        with ThreadPoolExecutor(max_workers=min(num_agents, 3)) as executor:
            futures = [executor.submit(vote_worker, i + 1) for i in range(num_agents)]

            for future in as_completed(futures):
                try:
                    future.result()  
                except Exception as e:
                    logging.error(f"Voting thread abnormality: {e}")

        voting_summary = self._analyze_voting_results(vote_results, chains_with_code)

        logging.info(f"Voting completed, winning chain: {voting_summary.get('winning_chain_id', 'None')}")
        return voting_summary

    def _format_chains_for_voting(self, chains_with_code: List[Dict[str, Any]]) -> str:
        chains_info_parts = []

        for chain_data in chains_with_code:
            chain_id = chain_data['chain_id']
            entities = chain_data['entities_with_code']

            chain_info = f"**{chain_id.upper()}:**\n"

            for i, entity in enumerate(entities):
                entity_info = f"Entity {i + 1}: {entity['entity_id']}\n"
                entity_info += f"Code:\n{entity.get('code', '# No code available')}\n\n"
                chain_info += entity_info

            chains_info_parts.append(chain_info)

        return "\n".join(chains_info_parts)

    def _parse_vote_result(self, response: str, agent_id: int) -> Dict[str, Any]:
        try:
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            vote_data = json.loads(response_text)
            vote_data['agent_id'] = agent_id

            if 'voted_chain_id' not in vote_data:
                raise ValueError("Missing voted_chain_id")

            logging.info(f"Agent {agent_id} success: {vote_data['voted_chain_id']}")
            return vote_data

        except Exception as e:
            logging.error(f"Agent {agent_id} fail: {e}")
            return {
                'agent_id': agent_id,
                'voted_chain_id': None,
                'confidence': 0,
                'reasoning': f'fail: {e}',
                'error': str(e)
            }

    def _analyze_voting_results(self, vote_results: List[Dict[str, Any]], chains_with_code: List[Dict[str, Any]]) -> \
            Dict[str, Any]:
        valid_votes = [v for v in vote_results if v.get('voted_chain_id') and 'error' not in v]
        invalid_votes = [v for v in vote_results if 'error' in v or not v.get('voted_chain_id')]

        logging.info(f"Valid votes: {len(valid_votes)}, Invalid votes: {len(invalid_votes)}")

        if not valid_votes:
            return {
                'success': False,
                'error': 'No valid votes received',
                'votes': vote_results,
                'winning_chain': None
            }

        vote_counts = Counter(v['voted_chain_id'] for v in valid_votes)

        winning_chain_id = vote_counts.most_common(1)[0][0]
        winning_votes = vote_counts[winning_chain_id]

        winning_chain_data = None
        for chain in chains_with_code:
            if chain['chain_id'] == winning_chain_id:
                winning_chain_data = chain
                break

        winning_confidences = [v.get('confidence', 0) for v in valid_votes if v['voted_chain_id'] == winning_chain_id]
        avg_confidence = sum(winning_confidences) / len(winning_confidences) if winning_confidences else 0

        voting_summary = {
            'success': True,
            'winning_chain_id': winning_chain_id,
            'winning_chain': winning_chain_data,
            'winning_votes': winning_votes,
            'total_valid_votes': len(valid_votes),
            'average_confidence': avg_confidence,
            'vote_distribution': dict(vote_counts),
            'all_votes': vote_results,
            'valid_votes': valid_votes,
            'invalid_votes': invalid_votes
        }

        logging.info(f"Voting result analysis completed:")
        logging.info(f"  Winning chain: {winning_chain_id}")

        return voting_summary

    def _generate_modification_plan(self, winning_chain: Dict[str, Any], issue_description: str, num_agents: int = 5, 
                                   instance_id: str = None, cache_timestamp: str = None) -> \
            Dict[str, Any]:
        """
        Stage 7: Generate multiple rounds of agent discussions to modify the plan

        Args:
            winning_chain: Winning positioning chain
            issue_description: Problem Description
            num_agents: Number of agents participating in the discussion
            instance_id: Instance ID for caching
            cache_timestamp: Cache timestamp

        Returns:
            Final modification plan
        """
        logging.info(f"=== Stage 7: Use {num_agents} agents to discuss modifying the plan ===")

        chain_info = self._format_chain_for_modification_discussion(winning_chain)

        #Each agent independently analyzes and modifies the position
        first_round_analyses = self._conduct_first_round_analysis(
            chain_info, issue_description, num_agents, instance_id, cache_timestamp
        )
        
        if instance_id and cache_timestamp:
            stage7_round1_data = {
                'first_round_analyses': first_round_analyses,
                'chain_info': chain_info,
                'num_agents': num_agents
            }
            self._save_stage_result(instance_id, 'stage_7_round1_analysis', stage7_round1_data, cache_timestamp)

        # debate
        second_round_analyses = self._conduct_second_round_analysis(
            chain_info, issue_description, first_round_analyses, instance_id, cache_timestamp
        )
        
        if instance_id and cache_timestamp:
            stage7_round2_data = {
                'second_round_analyses': second_round_analyses,
                'first_round_analyses': first_round_analyses
            }
            self._save_stage_result(instance_id, 'stage_7_round2_analysis', stage7_round2_data, cache_timestamp)

        # discriminator
        final_plan = self._conduct_final_discrimination(
            chain_info, issue_description, second_round_analyses, instance_id, cache_timestamp
        )
        
        if instance_id and cache_timestamp:
            stage7_final_data = {
                'final_plan': final_plan,
                'second_round_analyses': second_round_analyses
            }
            self._save_stage_result(instance_id, 'stage_7_final_plan', stage7_final_data, cache_timestamp)

        logging.info(
            f"Modification plan generation completed, including {len(final_plan.get('final_plan', {}).get('modifications', []))} modification steps"
        )
        return final_plan

    def _format_chain_for_modification_discussion(self, winning_chain: Dict[str, Any]) -> str:
        """Format winning chain information for modification discussion"""
        entities = winning_chain.get('entities_with_code', [])

        chain_info = f"**Winning Localization Chain ({winning_chain.get('chain_id', 'unknown')}):**\n\n"

        for i, entity in enumerate(entities):
            entity_info = f"**Entity {i + 1}: {entity['entity_id']}**\n"
            entity_info += f"Type: {entity.get('type', 'unknown')}\n"
            if entity.get('start_line') and entity.get('end_line'):
                entity_info += f"Lines: {entity['start_line']}-{entity['end_line']}\n"
            entity_info += f"Code:\n```\n{entity.get('code', '# No code available')}\n```\n\n"
            chain_info += entity_info

        return chain_info

    def _conduct_first_round_analysis(self, chain_info: str, issue_description: str, num_agents: int,
                                     instance_id: str = None, cache_timestamp: str = None) -> List[
        Dict[str, Any]]:
        first_round_results = []

        def analyze_worker(agent_id: int) -> Dict[str, Any]:
            try:
                prompt = MODIFICATION_LOCATION_PROMPT.format(
                    issue_description=issue_description,
                    chain_info=chain_info
                )

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert software engineer with deep experience in code analysis and debugging."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                response = self._call_llm_simple(
                    messages=messages,
                    temp=0.7,
                    max_tokens=3000
                )

                analysis = self._parse_modification_analysis(response, agent_id, "first_round")
                return analysis

            except Exception as e:
                logging.error(f"Agent {agent_id} fail: {e}")
                return {
                    'agent_id': agent_id,
                    'round': 'first_round',
                    'analysis': None,
                    'error': str(e)
                }

        with ThreadPoolExecutor(max_workers=min(num_agents, 3)) as executor:
            futures = [executor.submit(analyze_worker, i + 1) for i in range(num_agents)]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    first_round_results.append(result)
                except Exception as e:
                    logging.error(f"fail: {e}")

        return first_round_results
    
    def _conduct_second_round_analysis(self, chain_info: str, issue_description: str,
                                       first_round_analyses: List[Dict[str, Any]],
                                       instance_id: str = None, cache_timestamp: str = None) -> List[Dict[str, Any]]:
        second_round_results = []

        def prepare_other_agents_summary(current_agent_id: int) -> str:
            other_analyses = [a for a in first_round_analyses if
                              a.get('agent_id') != current_agent_id and a.get('analysis')]

            summary = ""
            for i, analysis in enumerate(other_analyses):
                agent_info = analysis.get('analysis', {})
                summary += f"\n**Agent {analysis.get('agent_id')} Analysis:**\n"
                summary += f"Strategy: {agent_info.get('overall_strategy', 'N/A')}\n"
                summary += f"Confidence: {agent_info.get('confidence', 'N/A')}\n"

                modifications = agent_info.get('modification_locations', [])
                if modifications:
                    summary += "Proposed modifications:\n"
                    for j, mod in enumerate(modifications): 
                        summary += f"  {j + 1}. {mod.get('entity_id', 'N/A')}: {mod.get('location_description', 'N/A')}\n"
                        summary += f"     Priority: {mod.get('priority', 'N/A')}, Type: {mod.get('modification_type', 'N/A')}\n"
                summary += "\n"

            return summary if summary else "No other valid analyses available."

        def analyze_worker_round2(agent_analysis: Dict[str, Any]) -> Dict[str, Any]:
            agent_id = agent_analysis.get('agent_id')

            try:
                your_initial_analysis = json.dumps(agent_analysis.get('analysis', {}), indent=2)
                other_agents_analyses = prepare_other_agents_summary(agent_id)

                prompt = COMPREHENSIVE_MODIFICATION_PROMPT.format(
                    issue_description=issue_description,
                    chain_info=chain_info,
                    your_initial_analysis=your_initial_analysis,
                    other_agents_analyses=other_agents_analyses
                )

                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert software engineer participating in a collaborative code review."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                response = self._call_llm_simple(
                    messages=messages,
                    temp=0.7,
                    max_tokens=6000
                )

                refined_analysis = self._parse_modification_analysis(response, agent_id, "second_round")
                return refined_analysis

            except Exception as e:
                logging.error(f"Agent {agent_id} fail: {e}")
                return {
                    'agent_id': agent_id,
                    'round': 'second_round',
                    'analysis': None,
                    'error': str(e)
                }

        valid_first_round = [a for a in first_round_analyses if a.get('analysis')]

        with ThreadPoolExecutor(max_workers=min(len(valid_first_round), 1)) as executor:
            futures = [executor.submit(analyze_worker_round2, analysis) for analysis in valid_first_round]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    second_round_results.append(result)
                except Exception as e:
                    logging.error(f"fail: {e}")

        valid_second_round = [r for r in second_round_results if r.get('analysis')]

        return second_round_results

    def _conduct_final_discrimination(self, chain_info: str, issue_description: str,
                                      second_round_analyses: List[Dict[str, Any]], 
                                      instance_id: str = None, cache_timestamp: str = None) -> Dict[str, Any]:
        all_agents_summary = ""
        valid_analyses = [a for a in second_round_analyses if a.get('analysis')]

        for analysis in valid_analyses:
            agent_id = analysis.get('agent_id')
            agent_analysis = analysis.get('analysis', {})

            all_agents_summary += f"\n**Agent {agent_id} Final Analysis:**\n"
            all_agents_summary += f"Overall Strategy: {agent_analysis.get('overall_strategy', 'N/A')}\n"
            all_agents_summary += f"Confidence: {agent_analysis.get('confidence', 'N/A')}\n"
            all_agents_summary += f"Key Insights: {agent_analysis.get('key_insights_learned', 'N/A')}\n"

            modifications = agent_analysis.get('refined_modification_locations', [])
            if modifications:
                all_agents_summary += "Proposed modifications:\n"
                for i, mod in enumerate(modifications):
                    all_agents_summary += f"  {i + 1}. Entity: {mod.get('entity_id', 'N/A')}\n"
                    all_agents_summary += f"     Location: {mod.get('location_description', 'N/A')}\n"
                    all_agents_summary += f"     Type: {mod.get('modification_type', 'N/A')}\n"
                    all_agents_summary += f"     Priority: {mod.get('priority', 'N/A')}\n"
                    all_agents_summary += f"     Reasoning: {mod.get('reasoning', 'N/A')}...\n"
            all_agents_summary += "\n"

        try:
            prompt = FINAL_DISCRIMINATOR_PROMPT.format(
                issue_description=issue_description,
                chain_info=chain_info,
                all_agents_analyses=all_agents_summary
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are the lead software architect responsible for making final technical decisions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = self._call_llm_simple(
                messages=messages,
                temp=0.3,  
                max_tokens=5000
            )

            final_plan = self._parse_final_plan(response)
            return final_plan

        except Exception as e:
            return {
                'success': False,
                'error': f'Final discrimination failed: {e}',
                'final_plan': {
                    'summary': 'Failed to generate plan',
                    'modifications': []
                }
            }

    def _parse_modification_analysis(self, response: str, agent_id: int, round_name: str) -> Dict[str, Any]:
        try:
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            analysis_data = json.loads(response_text)

            return {
                'agent_id': agent_id,
                'round': round_name,
                'analysis': analysis_data
            }

        except Exception as e:
            logging.error(f"Agent {agent_id} {round_name} fail: {e}")
            return {
                'agent_id': agent_id,
                'round': round_name,
                'analysis': None,
                'error': str(e)
            }

    def _parse_final_plan(self, response: str) -> Dict[str, Any]:
        try:
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            plan_data = json.loads(response_text)

            final_plan = plan_data.get('final_plan', {})
            if not final_plan.get('modifications'):
                raise ValueError("Final plan must contain modifications")

            return {
                'success': True,
                'final_plan': final_plan,
                'confidence': plan_data.get('confidence', 0),
                'expert_consensus': plan_data.get('expert_consensus', ''),
                'resolved_conflicts': plan_data.get('resolved_conflicts', '')
            }

        except Exception as e:
            logging.error(f"plan fail: {e}")
            return {
                'success': False,
                'error': f'Final plan parsing failed: {e}',
                'final_plan': {
                    'summary': 'Failed to parse plan',
                    'modifications': []
                }
            }

    def _extract_entity_groups(self, initial_entities: List[str], problem_statement: str) -> List[Dict[str, Any]]:
        entity_groups = []
        for i, initial_entity in enumerate(initial_entities):
            related_entities = self._extract_related_entities_for_initial_entity(
                initial_entity, problem_statement
            )
            entity_groups.append({
                'initial_entity': initial_entity,
                'related_entities': related_entities
            })
        return entity_groups

    def _generate_all_localization_chains(self, entity_groups: List[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        grouped_localization_chains = []
        all_chains = []

        for i, group in enumerate(entity_groups):
            initial_entity = group['initial_entity']
            localization_chains = self._generate_localization_chains(group['related_entities'])
            grouped_localization_chains.append({
                'initial_entity': initial_entity,
                'related_entities': group['related_entities'],
                'localization_chains': localization_chains,
                'chain_count': len(localization_chains)
            })

            for chain_info in localization_chains:
                if chain_info.get('chain') and len(chain_info['chain']) > 0:
                    all_chains.append({
                        'chain': chain_info['chain'],
                        'chain_length': chain_info['chain_length'],
                        'initial_entity': initial_entity,
                        'start_entity': chain_info['start_entity']
                    })

        return grouped_localization_chains, all_chains

    def _format_edit_agent_prompt(self, issue_description: str, modification_plan: Dict[str, Any],
                                  winning_chain: Dict[str, Any]) -> str:
        """
        Formats information output to the edit agent

        Args:
            issue_description: Problem Description
            modification_plan: Modification Plan
            winning_chain: Winning Localization Chain

        Returns:
            Formatted prompt string
        """
        issue_section = f"<issue>\n{issue_description}\n</issue>\n\n"

        plan_section = "<plan>\n"

        final_plan = modification_plan.get('final_plan', {})
        modifications = final_plan.get('modifications', [])

        for i, modification in enumerate(modifications):
            step = modification.get('step', i + 1)
            instruction = modification.get('instruction', 'No instruction provided')
            context = modification.get('context', 'No context provided')

            plan_section += f"***stage {step}***\n"
            plan_section += f"instruction: {instruction}\n"
            plan_section += f"context: {context}\n\n"

        plan_section += "</plan>\n\n"

        code_section = "<code>\n"

        entities_with_code = winning_chain.get('entities_with_code', [])

        file_entities = {}
        for entity in entities_with_code:
            entity_id = entity.get('entity_id', '')
            code = entity.get('code', '')

            if ':' in entity_id:
                file_path = entity_id.split(':')[0]
            else:
                file_path = entity_id

            if file_path not in file_entities:
                file_entities[file_path] = []

            file_entities[file_path].append({
                'entity_id': entity_id,
                'code': code
            })

        for file_path, entities in file_entities.items():
            code_section += f"{file_path}:\n"

            for entity in entities:
                code = entity['code']

                if code and code.strip():
                    formatted_code = self._clean_code_format(code)
                    code_section += formatted_code + "\n"
                else:
                    code_section += f"# No code available for {entity['entity_id']}\n"

            code_section += "\n" 

        code_section += "</code>"

        full_prompt = issue_section + plan_section + code_section

        logging.info(f"edit agent prompt formatting completed, total length: {len(full_prompt)}")

        return full_prompt

    def _clean_code_format(self, code: str) -> str:
        """
        Clean up the code format, remove the markdown tags but keep the original line numbers

        Args:
            code: A raw code string, possibly with markdown formatting and line numbers

        Returns:
            Cleaned code string
        """
        if not code or not code.strip():
            return "# No code content"

        cleaned_code = code.strip()

        if cleaned_code.startswith('```'):
            lines = cleaned_code.split('\n')
            start_idx = 1
            if len(lines) > 1:
                cleaned_code = '\n'.join(lines[start_idx:])

        if cleaned_code.endswith('```'):
            lines = cleaned_code.split('\n')
            if lines and lines[-1].strip() == '```':
                cleaned_code = '\n'.join(lines[:-1])

        lines = cleaned_code.split('\n')
        formatted_lines = []

        for line in lines:
            stripped_line = line.strip()
            if stripped_line and (
                    ('|' in line and line.split('|')[0].strip().isdigit()) or
                    (len(line) > 0 and line.split()[0].isdigit() and len(line.split()) > 1)
            ):
                if '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2 and parts[0].strip().isdigit():
                        line_num = parts[0].strip()
                        code_content = parts[1]
                        formatted_lines.append(f"{line_num:4s} {code_content}")
                    else:
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _ensure_cache_dir_exists(self):
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_file_path(self, instance_id: str, timestamp: str = None) -> str:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        instance_dir = os.path.join(self.cache_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        
        cache_file = os.path.join(instance_dir, f"{timestamp}_pipeline_cache.json")
        return cache_file
    
    def _save_stage_result(self, instance_id: str, stage_name: str, stage_data: Dict[str, Any], 
                          timestamp: str = None):
        if not self.enable_cache:
            return
            
        try:
            cache_file = self._get_cache_file_path(instance_id, timestamp)
            
            cache_data = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)

            cache_data[stage_name] = {
                'timestamp': datetime.now().isoformat(),
                'data': stage_data
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"Saved {stage_name} results to cache file: {cache_file}")
            
        except Exception as e:
            logging.error(f"Failed to save stage results: {e}")
    
    def _load_stage_result(self, instance_id: str, stage_name: str, timestamp: str = None) -> Optional[Dict[str, Any]]:
        if not self.enable_cache:
            return None
            
        try:
            cache_file = self._get_cache_file_path(instance_id, timestamp)
            
            if not os.path.exists(cache_file):
                return None
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if stage_name in cache_data:
                logging.info(f"Load {stage_name} result from cache: {cache_file}")
                return cache_data[stage_name]['data']
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to load stage result: {e}")
            return None
    
    def _list_cached_instances(self) -> List[str]:
        """List all cached instances"""
        try:
            if not os.path.exists(self.cache_dir):
                return []
            return [d for d in os.listdir(self.cache_dir) 
                   if os.path.isdir(os.path.join(self.cache_dir, d))]
        except Exception as e:
            logging.error(f"Failed to list cache instances: {e}")
            return []
    
    def _list_cached_files_for_instance(self, instance_id: str) -> List[str]:
        """List all cache files for the specified instance"""
        try:
            instance_dir = os.path.join(self.cache_dir, instance_id)
            if not os.path.exists(instance_dir):
                return []
            return [f for f in os.listdir(instance_dir) if f.endswith('_pipeline_cache.json')]
        except Exception as e:
            logging.error(f"Failed to list instance cache files: {e}")
            return []

