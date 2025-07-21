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

# 从context中提取entity的prompt模板（复用entity_search_scorer.py中的ENTITY_EXTRACTION_PROMPT）
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

# 从代码片段中提取相关entity的prompt模板
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

# 添加neighbor预筛选的prompt模板
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

# 节点选择的prompt模板
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

# 添加agent投票的prompt模板
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

# 添加第一轮修改位置判断的prompt模板
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

# 添加第二轮综合判断的prompt模板
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

        # 初始化定位链嵌入计算器
        self.chain_embedding = LocalizationChainEmbedding()
        
        # 缓存相关配置
        self.cache_dir = "/entity_pipeline_cache"
        self.enable_cache = True
        self._ensure_cache_dir_exists()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url="",
            api_key="",
            timeout=60.0
        )

    def _call_llm_simple(self, messages: List[Dict], temp: float = 0.1, max_tokens: int = 1000) -> str:
        """
        简单的LLM调用函数，复用entity_search_scorer.py中的逻辑

        Args:
            messages: 消息列表
            temp: 温度参数
            max_tokens: 最大token数

        Returns:
            模型响应文本
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
            logging.error(f"LLM调用失败: {e}")
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

        # 存储issue描述供LLM使用
        self._current_issue_description = instance_data['problem_statement']
        
        # 生成缓存时间戳
        cache_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        instance_id = instance_data.get('instance_id', 'unknown')
        
        logging.info(f"缓存时间戳: {cache_timestamp}")

        try:
            # Stage 1: Extract initial entities from context
            logging.info("开始阶段1: 从context中提取初始entities")
            initial_entities = self._extract_initial_entities(context, instance_data['problem_statement'],
                                                              max_initial_entities)
            logging.info(f"阶段1完成，提取到 {len(initial_entities)} 个初始entities: {initial_entities}")
            
            # 保存Stage 1结果
            stage1_data = {
                'initial_entities': initial_entities,
                'context': context,
                'max_initial_entities': max_initial_entities,
                'context_length': len(context),
                'problem_statement_length': len(instance_data.get('problem_statement', ''))
            }
            self._save_stage_result(instance_id, 'stage_1_initial_entities', stage1_data, cache_timestamp)

            if not initial_entities:
                logging.warning("未提取到任何初始entities，流程结束")
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
            logging.info("开始阶段2: 为每个初始entity搜索相关entities")
            entity_groups = []
            for i, initial_entity in enumerate(initial_entities):
                logging.info(f"处理初始entity {i + 1}/{len(initial_entities)}: '{initial_entity}'")
                related_entities = self._extract_related_entities_for_initial_entity(
                    initial_entity, instance_data['problem_statement']
                )
                entity_groups.append({
                    'initial_entity': initial_entity,
                    'related_entities': related_entities
                })
                logging.info(f"为 '{initial_entity}' 找到 {len(related_entities)} 个相关entities")

            total_related_entities = sum(len(group['related_entities']) for group in entity_groups)
            logging.info(f"阶段2完成，总共找到 {total_related_entities} 个相关entities")
            
            # 保存Stage 2结果
            stage2_data = {
                'entity_groups': entity_groups,
                'total_related_entities': total_related_entities
            }
            self._save_stage_result(instance_id, 'stage_2_related_entities', stage2_data, cache_timestamp)

            # Stage 3: Generate localization chains for each related entity, grouped by initial entity (并行版本)
            logging.info("开始阶段3: 为相关entities生成定位链 (并行处理)")
            grouped_localization_chains = []
            all_chains = []  # 收集所有定位链用于后续选择
            total_chains_generated = 0
            results_lock = threading.Lock()

            def process_group_worker(group_index: int, group: Dict[str, Any]) -> Dict[str, Any]:
                """单个group的定位链生成工作函数"""
                initial_entity = group['initial_entity']
                logging.info(f"开始处理group {group_index + 1}/{len(entity_groups)} - 初始entity: '{initial_entity}'")
                
                try:
                    localization_chains = self._generate_localization_chains(group['related_entities'])
                    
                    group_result = {
                        'initial_entity': initial_entity,
                        'related_entities': group['related_entities'],
                        'localization_chains': localization_chains,
                        'chain_count': len(localization_chains),
                        'group_index': group_index
                    }
                    
                    # 收集该group的有效定位链
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
                    
                    logging.info(f"Group {group_index + 1} ('{initial_entity}') 处理完成，生成了 {len(localization_chains)} 条定位链")
                    return group_result
                    
                except Exception as e:
                    logging.error(f"Group {group_index + 1} ('{initial_entity}') 处理失败: {e}")
                    return {
                        'initial_entity': initial_entity,
                        'related_entities': group['related_entities'],
                        'localization_chains': [],
                        'chain_count': 0,
                        'group_index': group_index,
                        'valid_chains': [],
                        'error': str(e)
                    }

            # 使用线程池并行处理所有groups
            max_workers = min(len(entity_groups), 5)  # 限制并发数避免过度占用资源
            logging.info(f"启动 {max_workers} 个工作线程进行并行group处理")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有group的处理任务
                futures = [
                    executor.submit(process_group_worker, i, group)
                    for i, group in enumerate(entity_groups)
                ]

                # 收集完成的结果
                completed_results = []
                for future in as_completed(futures):
                    try:
                        group_result = future.result()
                        with results_lock:
                            completed_results.append(group_result)
                    except Exception as e:
                        logging.error(f"Group处理线程异常: {e}")

            # 按原始顺序排序结果
            completed_results.sort(key=lambda x: x['group_index'])
            
            # 构建最终结果
            for group_result in completed_results:
                # 移除辅助字段，保持原有格式
                final_group_result = {
                    'initial_entity': group_result['initial_entity'],
                    'related_entities': group_result['related_entities'],
                    'localization_chains': group_result['localization_chains'],
                    'chain_count': group_result['chain_count']
                }
                grouped_localization_chains.append(final_group_result)
                
                # 收集所有有效的定位链
                all_chains.extend(group_result.get('valid_chains', []))
                total_chains_generated += group_result['chain_count']

            logging.info(f"阶段3并行处理完成，总共生成 {total_chains_generated} 条定位链")
            for i, chain in enumerate(all_chains, 1):
                logging.info("Chain %d: %s", i, chain)
            
            # 保存Stage 3结果
            stage3_data = {
                'grouped_localization_chains': grouped_localization_chains,
                'all_chains': all_chains,
                'total_chains_generated': total_chains_generated
            }
            self._save_stage_result(instance_id, 'stage_3_localization_chains', stage3_data, cache_timestamp)

            # Stage 4: 使用embedding选择多样化的定位链
            logging.info("开始阶段4: 使用embedding选择多样化的定位链")
            selected_chains = self._select_diverse_chains(all_chains)
            
            # 保存Stage 4结果
            stage4_data = {
                'selected_chains': selected_chains,
                'selected_chains_count': len(selected_chains)
            }
            self._save_stage_result(instance_id, 'stage_4_diverse_chains', stage4_data, cache_timestamp)

            # Stage 5: 为选择的定位链添加代码信息
            logging.info("开始阶段5: 为选择的定位链添加代码信息")
            chains_with_code = self._add_code_to_chains(selected_chains)
            
            # 保存Stage 5结果
            stage5_data = {
                'chains_with_code': chains_with_code,
                'chains_with_code_count': len(chains_with_code)
            }
            self._save_stage_result(instance_id, 'stage_5_chains_with_code', stage5_data, cache_timestamp)

            # Stage 6: 使用多个agent对定位链进行投票
            logging.info("开始阶段6: 使用多个agent对定位链进行投票")
            voting_result = self._vote_on_chains(chains_with_code, instance_data['problem_statement'])
            
            # 保存Stage 6结果（包括winning chain的信息）
            stage6_data = {
                'voting_result': voting_result,
                'chains_with_code': chains_with_code,  # 保存用于投票的链信息
                'winning_chain': voting_result.get('winning_chain'),
                'winning_chain_id': voting_result.get('winning_chain_id')
            }
            self._save_stage_result(instance_id, 'stage_6_voting_result', stage6_data, cache_timestamp)

            if not voting_result.get('success') or not voting_result.get('winning_chain'):
                return self._create_error_result(instance_data, context, 'No winning chain found')

            # Stage 7: 多轮agent讨论生成修改plan
            logging.info("开始阶段7: 多轮agent讨论生成修改plan")
            modification_plan = self._generate_modification_plan(
                voting_result['winning_chain'],
                instance_data['problem_statement'],
                5,  # num_agents
                instance_id,
                cache_timestamp
            )

            # Stage 8: 格式化输出给edit agent的信息
            logging.info("开始阶段8: 格式化输出给edit agent的信息")
            edit_agent_prompt = self._format_edit_agent_prompt(
                instance_data['problem_statement'],
                modification_plan,
                voting_result['winning_chain']
            )
            
            # 保存Stage 8结果
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
                'edit_agent_prompt': edit_agent_prompt,  # 新添加的格式化输出
                'total_chains': len(all_chains),
                'metadata': {
                    'repo': instance_data['repo'],
                    'base_commit': instance_data['base_commit'],
                    'problem_statement': instance_data['problem_statement']
                }
            }

            logging.info("=" * 80)
            logging.info("=== Entity Localization Pipeline 完成 ===")
            logging.info(f"最终结果汇总:")
            logging.info(f"  - 初始entities: {len(initial_entities)}")
            logging.info(f"  - 总定位链数: {len(all_chains)}")
            logging.info(f"  - 选择的定位链数: {len(selected_chains)}")
            logging.info(f"  - 最终获胜链: {voting_result.get('winning_chain_id', 'None')}")
            logging.info(f"  - 修改plan步骤数: {len(modification_plan.get('final_plan', {}).get('modifications', []))}")
            logging.info(f"  - edit agent prompt长度: {len(edit_agent_prompt)}")
            logging.info("=" * 80)

            # return result
            return edit_agent_prompt

        finally:
            # Cleanup
            reset_current_issue()
            logging.info("Current issue cleanup completed")

    def _create_error_result(self, instance_data: Dict[str, Any], context: str, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
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
        logging.info(f"输入context长度: {len(context)}")
        logging.info(f"输入issue描述长度: {len(issue_description)}")
        logging.info(f"最大entity数量: {max_entities}")

        prompt = ENTITY_EXTRACTION_PROMPT.format(
            context=context,
            issue_description=issue_description,
            max_entities=max_entities
        )

        logging.info(f"构建的entity提取prompt长度: {len(prompt)}")
        messages = [{"role": "user", "content": prompt}]

        try:
            logging.info("调用LLM进行初始entity提取...")
            response = self._call_llm_simple(
                messages=messages,
                temp=0.7,
                max_tokens=1000
            )

            logging.info(f"LLM响应长度: {len(response)}")
            logging.info(f"LLM原始响应: {response[:300]}...")

            # 解析JSON响应
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                logging.info("移除了开头的```json标记")
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                logging.info("移除了结尾的```标记")

            logging.info(f"清理后的响应: {response_text}")
            entities = json.loads(response_text)

            # 验证返回的是列表且包含字符串，并确保数量符合要求
            if isinstance(entities, list) and all(isinstance(entity, str) for entity in entities):
                # 确保数量不超过max_entities（模型可能返回少于max_entities的数量）
                entities = entities[:max_entities]
                logging.info(f"成功解析JSON，提取到 {len(entities)} 个entities (要求最多{max_entities}个)")
                logging.info(f"成功提取初始entities: {entities}")
                return entities
            else:
                logging.warning(f"大模型返回格式不正确，返回类型: {type(entities)}")
                return []

        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}，原始响应: {response}")
            return []
        except Exception as e:
            logging.error(f"初始entity提取失败: {e}")
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
        logging.info(f"开始搜索包含 '{initial_entity}' 的代码片段...")
        code_snippets = self._search_code_snippets_for_entity(initial_entity)

        if not code_snippets or len(code_snippets.strip()) == 0:
            logging.warning(f"No code snippets found for entity '{initial_entity}'")
            return []

        logging.info(f"找到代码片段，长度: {len(code_snippets)}")

        # Extract 4 related entities from the code snippets
        logging.info(f"从代码片段中提取相关entities...")
        related_entities = self._extract_entities_from_code_snippets(code_snippets, problem_statement)

        logging.info(f"Found {len(related_entities)} related entities for '{initial_entity}'")
        for i, entity in enumerate(related_entities):
            logging.info(
                f"  相关entity {i + 1}: {entity.get('entity_id', 'unknown')} ({entity.get('entity_type', 'unknown')})"
            )
            logging.info(f"    原因: {entity.get('relevance_reason', 'No reason provided')}")

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

        logging.info(f"搜索词列表: {search_terms}")

        # Search for code snippets
        logging.info("调用search_code_snippets进行搜索...")
        code_snippets = search_code_snippets(
            search_terms=search_terms,
            file_path_or_pattern="**/*.py"
        )

        logging.info(f"搜索完成，找到代码片段长度: {len(code_snippets)}")
        if code_snippets:
            logging.info(f"代码片段预览: {code_snippets[:200]}...")

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
        logging.info("开始从代码片段中提取entities...")

        prompt = CODE_SNIPPET_ENTITY_EXTRACTION_PROMPT.format(
            code_snippets=code_snippets,
            problem_statement=problem_statement
        )

        logging.info(f"构建的entity提取prompt长度: {len(prompt)}")

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
            logging.info("调用LLM进行entity提取...")
            response = self.client.chat.completions.create(
                model="deepseek-v3",
                messages=messages,
                temperature=0.7,
            )

            entities_text = response.choices[0].message.content
            logging.info(f"LLM响应长度: {len(entities_text)}")
            logging.info(f"LLM原始响应: {entities_text[:300]}...")

            entities = self._parse_extracted_entities(entities_text)

            logging.info(f"成功提取 {len(entities)} 个entities from code snippets")
            return entities

        except Exception as e:
            logging.error(f"Entity extraction from code snippets failed: {e}")
            return []

    def _parse_extracted_entities(self, entities_text: str) -> List[Dict[str, Any]]:
        """Parse extracted entities from LLM response."""
        logging.info("开始解析LLM返回的entities...")
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*?\]', entities_text, re.DOTALL)
            if json_match:
                entities_json = json_match.group()
                logging.info(f"提取到JSON字符串: {entities_json[:200]}...")
                entities = json.loads(entities_json)

                # Validate and clean entities
                validated_entities = []
                for i, entity in enumerate(entities[:4]):  # Ensure max 4 entities
                    if 'entity_id' in entity and 'entity_type' in entity:
                        validated_entities.append(entity)
                        logging.info(f"验证entity {i + 1}: {entity['entity_id']} ({entity['entity_type']})")
                    else:
                        logging.warning(f"跳过无效entity {i + 1}: {entity}")

                logging.info(f"验证完成，有效entities: {len(validated_entities)}")
                return validated_entities
            else:
                logging.warning("在响应中未找到JSON格式数据")
        except json.JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}")
        except Exception as e:
            logging.error(f"Failed to parse entities: {e}")

        return []

    def _select_next_node_with_llm(self, current_entity: str, neighbors: List[str],
                                   issue_description: str, entity_searcher, depth: int) -> Dict[str, Any]:
        """
        使用LLM选择下一个最有希望的探索节点

        Args:
            current_entity: 当前实体ID
            neighbors: 邻居节点列表
            issue_description: 问题描述
            entity_searcher: 实体搜索器
            depth: 当前深度

        Returns:
            包含选择决策的字典
        """
        logging.info(f"使用LLM选择下一个节点，当前entity: {current_entity}, 深度: {depth}, 邻居数量: {len(neighbors)}")

        if not neighbors:
            logging.info("没有可用邻居，停止探索")
            return {"should_continue": False, "selected_neighbor": None, "reasoning": "No neighbors available"}

        try:
            # 获取当前实体信息
            current_entity_data = entity_searcher.get_node_data([current_entity])[0]
            current_entity_type = current_entity_data['type']

            logging.info(f"当前实体类型: {current_entity_type}")

            # 构建邻居信息
            neighbor_info_list = []
            for i, neighbor in enumerate(neighbors[:10]):  # 限制最多10个邻居避免prompt过长
                try:
                    neighbor_data = entity_searcher.get_node_data([neighbor])[0]
                    neighbor_info = f"- {neighbor} (Type: {neighbor_data['type']})"
                    # 如果有代码信息，添加简短描述
                    if 'code' in neighbor_data and neighbor_data['code']:
                        code_preview = neighbor_data['code'][:200] + "..." if len(neighbor_data['code']) > 200 else \
                            neighbor_data['code']
                        neighbor_info += f"\n  Code preview: {code_preview}"
                    neighbor_info_list.append(neighbor_info)
                    logging.info(f"  邻居 {i + 1}: {neighbor} ({neighbor_data['type']})")
                except:
                    neighbor_info_list.append(f"- {neighbor} (Type: unknown)")
                    logging.warning(f"  邻居 {i + 1}: {neighbor} (获取信息失败)")

            neighbor_info = "\n".join(neighbor_info_list)

            # 构建prompt
            prompt = NODE_SELECTION_PROMPT.format(
                issue_description=issue_description,
                current_entity=current_entity,
                current_entity_type=current_entity_type,
                depth=depth,
                neighbor_info=neighbor_info
            )

            logging.info(f"构建的节点选择prompt长度: {len(prompt)}")
            messages = [{"role": "user", "content": prompt}]

            logging.info("调用LLM进行节点选择...")
            response = self._call_llm_simple(
                messages=messages,
                temp=0.7,
                max_tokens=1000
            )

            logging.info(f"LLM节点选择响应: {response[:200]}...")

            # 解析JSON响应
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            decision = json.loads(response_text)

            # 验证响应格式
            if isinstance(decision, dict) and 'should_continue' in decision:
                should_continue = decision.get('should_continue', False)
                selected_neighbor = decision.get('selected_neighbor')
                reasoning = decision.get('reasoning', 'No reasoning provided')
                logging.info(f"LLM决策: 继续探索={should_continue}, 选中邻居={selected_neighbor}")
                logging.info(f"LLM推理: {reasoning}")
                return decision
            else:
                logging.warning("LLM返回格式不正确，使用fallback逻辑")
                return self._fallback_node_selection(current_entity, neighbors, entity_searcher, depth)

        except Exception as e:
            logging.error(f"LLM节点选择失败: {e}，使用fallback逻辑")
            return self._fallback_node_selection(current_entity, neighbors, entity_searcher, depth)

    def _prefilter_neighbors_with_llm(self, current_entity: str, all_neighbors: List[Dict],
                                      issue_description: str, entity_searcher, depth: int,
                                      max_selection: int = 10) -> List[str]:
        """
        使用LLM预筛选neighbors，选出最相关和多样化的邻居实体

        Args:
            current_entity: 当前实体ID
            all_neighbors: 所有邻居实体信息列表
            issue_description: 问题描述
            entity_searcher: 实体搜索器
            depth: 当前深度
            max_selection: 最大选择数量

        Returns:
            筛选后的邻居实体ID列表
        """
        if len(all_neighbors) <= max_selection:
            # 如果邻居数量不超过最大选择数，直接返回所有邻居
            return [n['entity_id'] for n in all_neighbors]

        logging.info(f"开始预筛选neighbors，总数: {len(all_neighbors)}, 目标选择: {max_selection}")

        try:
            # 获取当前实体信息
            current_entity_data = entity_searcher.get_node_data([current_entity])[0]
            current_entity_type = current_entity_data['type']

            # 构建邻居列表信息（只包含entity ID和基本类型信息）
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

            # 构建预筛选prompt
            prompt = NEIGHBOR_PREFILTERING_PROMPT.format(
                issue_description=issue_description,
                current_entity=current_entity,
                current_entity_type=current_entity_type,
                depth=depth,
                total_count=len(all_neighbors),
                neighbor_list=neighbor_list,
                max_selection=max_selection
            )

            logging.info(f"构建的neighbor预筛选prompt长度: {len(prompt)}")
            messages = [{"role": "user", "content": prompt}]

            logging.info("调用LLM进行neighbor预筛选...")
            response = self._call_llm_simple(
                messages=messages,
                temp=0.7,
                max_tokens=1000
            )

            logging.info(f"LLM预筛选响应: {response[:200]}...")

            # 解析JSON响应
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            selection_result = json.loads(response_text)

            # 验证响应格式并提取选中的邻居
            if isinstance(selection_result, dict) and 'selected_neighbors' in selection_result:
                selected_neighbors = selection_result['selected_neighbors']
                reasoning = selection_result.get('selection_reasoning', 'No reasoning provided')
                diversity = selection_result.get('diversity_considerations', 'No diversity info')

                # 验证选中的邻居是否都在原始列表中
                available_neighbor_ids = [n['entity_id'] for n in all_neighbors]
                valid_selected = [n for n in selected_neighbors if n in available_neighbor_ids]

                logging.info(f"LLM选择了 {len(valid_selected)} 个neighbors")
                logging.info(f"选择原因: {reasoning}")
                logging.info(f"多样性考虑: {diversity}")
                logging.info(f"选中的neighbors: {valid_selected}")

                return valid_selected
            else:
                logging.warning("LLM预筛选返回格式不正确，使用fallback策略")
                return self._fallback_neighbor_prefiltering(all_neighbors, max_selection)

        except Exception as e:
            logging.error(f"LLM neighbor预筛选失败: {e}，使用fallback策略")
            return self._fallback_neighbor_prefiltering(all_neighbors, max_selection)

    def _fallback_neighbor_prefiltering(self, all_neighbors: List[Dict], max_selection: int) -> List[str]:
        """
        Fallback neighbor预筛选策略

        Args:
            all_neighbors: 所有邻居实体信息列表
            max_selection: 最大选择数量

        Returns:
            筛选后的邻居实体ID列表
        """
        logging.info("使用fallback neighbor预筛选策略")

        # 简单策略：尽量选择不同文件的entities，优先选择functions和classes
        selected = []
        seen_files = set()

        # 按优先级排序：function > class > file
        type_priority = {'function': 0, 'class': 1, 'file': 2}

        # 首先尝试选择不同文件的entities
        for neighbor_info in all_neighbors:
            if len(selected) >= max_selection:
                break

            neighbor_id = neighbor_info['entity_id']
            file_path = neighbor_id.split(':')[0] if ':' in neighbor_id else neighbor_id

            # 如果是新文件，优先选择
            if file_path not in seen_files:
                selected.append(neighbor_id)
                seen_files.add(file_path)

        # 如果还没有达到最大数量，继续选择剩余的
        for neighbor_info in all_neighbors:
            if len(selected) >= max_selection:
                break

            neighbor_id = neighbor_info['entity_id']
            if neighbor_id not in selected:
                selected.append(neighbor_id)

        logging.info(f"Fallback策略选择了 {len(selected)} 个neighbors: {selected}")
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
        logging.info(f"开始DFS遍历，起始entity: {start_entity}, 最大深度: {max_depth}")

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

            logging.info(f"DFS访问节点: {current_entity}, 深度: {depth}, 路径长度: {len(current_path)}")

            if depth >= max_depth or current_entity in visited:
                if depth >= max_depth:
                    logging.info(f"达到最大深度 {max_depth}，停止探索")
                if current_entity in visited:
                    logging.info(f"节点 {current_entity} 已访问过，跳过")
                return False

            visited.add(current_entity)

            # Add current step to path - only save entity_id
            current_path.append(current_entity)

            # 收集所有可用的邻居
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

            logging.info(f"找到 {len(all_neighbors)} 个未访问的邻居")

            # 如果没有未访问的邻居，这是一个端点
            if not all_neighbors:
                if len(current_path) > len(best_chain):
                    best_chain = current_path.copy()
                    logging.info(f"更新最佳路径，长度: {len(best_chain)}")
                return True

            # 步骤1：使用LLM预筛选neighbors（如果邻居太多）
            if len(all_neighbors) > 10:
                logging.info(f"邻居数量({len(all_neighbors)})超过10，开始预筛选")
                prefiltered_neighbors = self._prefilter_neighbors_with_llm(
                    current_entity, all_neighbors, issue_description, entity_searcher, depth, max_selection=10
                )
            else:
                prefiltered_neighbors = [n['entity_id'] for n in all_neighbors]
                logging.info(f"邻居数量({len(all_neighbors)})不超过10，无需预筛选")

            logging.info(f"预筛选后的邻居数量: {len(prefiltered_neighbors)}")

            # 步骤2：使用LLM从预筛选的neighbors中选择最佳的一个
            decision = self._select_next_node_with_llm(
                current_entity, prefiltered_neighbors, issue_description, entity_searcher, depth
            )

            if not decision.get('should_continue', False):
                # LLM决定停止探索，将当前路径作为一个候选链
                if len(current_path) > len(best_chain):
                    best_chain = current_path.copy()
                    logging.info(f"LLM决定停止，更新最佳路径，长度: {len(best_chain)}")
                return True

            selected_neighbor = decision.get('selected_neighbor')
            if selected_neighbor and selected_neighbor in prefiltered_neighbors:
                logging.info(f"继续探索选中邻居: {selected_neighbor}")
                # 递归调用选中的邻居 (不再保存边信息)
                if dfs(selected_neighbor, depth + 1, current_path.copy()):
                    return True

            # 如果选中的邻居探索失败，尝试其他预筛选的邻居（限制数量避免过度探索）
            backup_count = 0
            for neighbor in prefiltered_neighbors[:3]:  # 最多尝试3个其他邻居
                if neighbor != selected_neighbor and neighbor not in visited:
                    backup_count += 1
                    logging.info(f"尝试备选邻居 {backup_count}: {neighbor}")

                    # 不再保存边信息，直接递归调用
                    if dfs(neighbor, depth + 1, current_path.copy()):
                        return True

            return False

        # 在开始DFS之前存储issue描述
        issue_description = self._current_issue_description

        # Start DFS
        logging.info("启动DFS搜索...")
        dfs(start_entity, 0, [])

        logging.info(f"DFS完成，最佳链长度: {len(best_chain)}")
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
        logging.info(f"为 {len(entities)} 个entities生成定位链")

        if not entities:
            logging.warning("没有entities用于生成定位链")
            return []

        graph = get_graph()
        entity_searcher = get_graph_entity_searcher()
        dependency_searcher = get_graph_dependency_searcher()

        logging.info("图对象获取完成")

        all_chains = []
        chain_lock = threading.Lock()

        def generate_chain_worker(entity_index: int, entity: Dict[str, Any]) -> Dict[str, Any]:
            """
            单个entity的定位链生成工作函数

            Args:
                entity_index: entity在列表中的索引
                entity: entity信息字典

            Returns:
                包含定位链信息的字典
            """
            entity_id = entity['entity_id']
            logging.info(f"处理entity {entity_index + 1}/{len(entities)}: {entity_id}")

            try:
                if not entity_searcher.has_node(entity_id):
                    logging.warning(f"Entity {entity_id} not found in graph")
                    return {
                        'start_entity': entity,
                        'chain': [],
                        'chain_length': 0,
                        'error': 'Entity not found in graph'
                    }

                logging.info(f"在图中找到entity {entity_id}，开始DFS遍历...")

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

                logging.info(f"Entity {entity_id} DFS完成，生成链长度: {len(chain)}")
                if chain:
                    logging.info(f"  简化定位链: {chain}")

                return chain_info

            except Exception as e:
                logging.error(f"Entity {entity_id} 定位链生成失败: {e}")
                return {
                    'start_entity': entity,
                    'chain': [],
                    'chain_length': 0,
                    'error': f'Chain generation failed: {e}'
                }

        # 使用线程池并行生成定位链
        max_workers = min(len(entities), 5)  # 限制并发数避免过度占用资源
        logging.info(f"启动 {max_workers} 个工作线程进行并行定位链生成")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有entity的生成任务
            futures = [
                executor.submit(generate_chain_worker, i, entity)
                for i, entity in enumerate(entities)
            ]

            # 收集完成的结果
            for future in as_completed(futures):
                try:
                    chain_info = future.result()
                    with chain_lock:
                        all_chains.append(chain_info)
                except Exception as e:
                    logging.error(f"定位链生成线程异常: {e}")
                    # 添加一个空的chain_info避免丢失计数
                    with chain_lock:
                        all_chains.append({
                            'start_entity': {'entity_id': 'unknown'},
                            'chain': [],
                            'chain_length': 0,
                            'error': f'Thread execution failed: {e}'
                        })

        # 按原始顺序排序结果（因为as_completed可能改变顺序）
        # 通过start_entity的entity_id来匹配原始顺序
        entity_ids_order = [entity['entity_id'] for entity in entities]
        ordered_chains = []

        for entity_id in entity_ids_order:
            # 找到对应的chain_info
            matching_chain = None
            for chain_info in all_chains:
                if chain_info['start_entity'].get('entity_id') == entity_id:
                    matching_chain = chain_info
                    break

            if matching_chain:
                ordered_chains.append(matching_chain)
            else:
                # 如果没有找到匹配的，添加一个错误记录
                logging.warning(f"未找到entity {entity_id} 的定位链结果")
                ordered_chains.append({
                    'start_entity': {'entity_id': entity_id},
                    'chain': [],
                    'chain_length': 0,
                    'error': 'Result not found after parallel execution'
                })

        logging.info(f"所有定位链生成完成，总数: {len(ordered_chains)}")

        # 统计成功和失败的数量
        successful_chains = [c for c in ordered_chains if c.get('chain') and len(c['chain']) > 0]
        failed_chains = [c for c in ordered_chains if c.get('error')]

        logging.info(f"成功生成定位链: {len(successful_chains)}")
        logging.info(f"失败的定位链: {len(failed_chains)}")

        return ordered_chains

    def _select_diverse_chains(self, all_chains: List[Dict[str, Any]], max_selected: int = 6) -> List[Dict[str, Any]]:
        """
        Stage 4: 使用embedding选择多样化的定位链（基于最长链）

        Args:
            all_chains: 所有生成的定位链
            max_selected: 最大选择数量（包括最长链）

        Returns:
            选择的定位链列表
        """
        logging.info(f"=== Stage 4: 选择多样化的定位链（基于最长链） ===")
        logging.info(f"输入定位链总数: {len(all_chains)}")

        if not all_chains:
            logging.warning("没有可用的定位链")
            return []

        # 检查空链数量
        non_empty_chains = [chain for chain in all_chains if chain.get('chain') and len(chain['chain']) > 0]
        logging.info(f"非空定位链数量: {len(non_empty_chains)}")

        if len(non_empty_chains) <= max_selected:
            logging.info(f"非空定位链数量({len(non_empty_chains)})不超过最大选择数量({max_selected})，全部返回")
            return non_empty_chains

        # 提取纯定位链用于embedding计算
        chains_for_embedding = [chain_info['chain'] for chain_info in all_chains]

        try:
            # 使用embedding选择多样化的链（包含空链过滤和去重）
            selected_indices, similarity_scores = self.chain_embedding.select_diverse_chains(
                chains_for_embedding, k=max_selected - 1  # -1因为已包含最长链
            )

            if not selected_indices:
                logging.warning("embedding选择未返回任何索引，使用fallback策略")
                # fallback: 按长度降序选择
                sorted_chains = sorted(non_empty_chains, key=lambda x: x['chain_length'], reverse=True)
                return sorted_chains[:max_selected]

            # 构建选择的定位链结果
            selected_chains = []
            for i, idx in enumerate(selected_indices):
                if idx >= len(all_chains):
                    logging.warning(f"索引 {idx} 超出范围，跳过")
                    continue

                chain_info = deepcopy(all_chains[idx])
                chain_info['selection_rank'] = i + 1
                chain_info['similarity_to_longest'] = similarity_scores[i] if i < len(similarity_scores) else 1.0
                chain_info['is_longest'] = (i == 0)  # 第一个总是最长的
                selected_chains.append(chain_info)

            logging.info(f"成功选择 {len(selected_chains)} 条多样化定位链")
            for i, chain in enumerate(selected_chains):
                logging.info(f"  选择链 {i + 1}: 长度={chain['chain_length']}, "
                             f"相似度={chain['similarity_to_longest']:.3f}, "
                             f"是否最长={chain['is_longest']}")
                logging.info(f"    链内容: {chain['chain']}")

            return selected_chains

        except Exception as e:
            logging.error(f"定位链选择失败: {e}，返回按长度排序的链")
            # 如果embedding选择失败，返回长度最长的几条链
            sorted_chains = sorted(non_empty_chains, key=lambda x: x['chain_length'], reverse=True)
            return sorted_chains[:max_selected]

    def _fallback_node_selection(self, current_entity: str, neighbors: List[str], entity_searcher, depth: int) -> Dict[
        str, Any]:
        """
        LLM节点选择的fallback逻辑

        Args:
            current_entity: 当前实体ID
            neighbors: 邻居节点列表
            entity_searcher: 实体搜索器
            depth: 当前深度

        Returns:
            包含选择决策的字典
        """
        logging.info(f"使用fallback逻辑选择节点，当前深度: {depth}")

        if not neighbors:
            return {"should_continue": False, "selected_neighbor": None, "reasoning": "No neighbors available"}

        # 简单的fallback策略：
        # 1. 如果深度太深，停止探索
        if depth >= self.max_depth - 1:
            return {
                "should_continue": False,
                "selected_neighbor": None,
                "reasoning": f"Reached maximum depth {self.max_depth}"
            }

        # 2. 随机选择一个邻居继续探索
        selected_neighbor = neighbors[0]  # 选择第一个邻居

        return {
            "should_continue": True,
            "selected_neighbor": selected_neighbor,
            "reasoning": f"Fallback selection: chose first neighbor {selected_neighbor}"
        }

    def _add_code_to_chains(self, selected_chains: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Stage 5: 为选择的定位链添加代码信息

        Args:
            selected_chains: 选择的定位链列表

        Returns:
            包含代码信息的定位链列表
        """
        logging.info(f"=== Stage 5: 为 {len(selected_chains)} 条定位链添加代码信息 ===")

        chains_with_code = []
        entity_searcher = get_graph_entity_searcher()
        MAX_CODE_LENGTH = 10000  # 设置代码长度限制

        for i, chain_info in enumerate(selected_chains):
            chain_id = f"chain_{i + 1}"
            chain = chain_info['chain']

            logging.info(f"处理链 {chain_id}，长度: {len(chain)}")

            # 为链中的每个实体获取代码
            entities_with_code = []
            for j, entity_id in enumerate(chain):
                logging.info(f"  获取实体 {j + 1}/{len(chain)}: {entity_id}")

                try:
                    if entity_searcher.has_node(entity_id):
                        entity_data = entity_searcher.get_node_data([entity_id], return_code_content=True)[0]
                        code_content = entity_data.get('code_content', '')

                        # 检查代码长度，如果超过限制则不保留代码
                        if len(code_content) > MAX_CODE_LENGTH:
                            logging.info(
                                f"    实体代码长度 {len(code_content)} 超过限制 {MAX_CODE_LENGTH}，仅保留entity名字")
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
                            f"    成功获取代码，长度: {len(entity_with_code['code'])} (原始长度: {len(code_content)})")
                    else:
                        logging.warning(f"    实体 {entity_id} 在图中不存在")
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
                    logging.error(f"    获取实体 {entity_id} 代码失败: {e}")
                    entities_with_code.append({
                        'entity_id': entity_id,
                        'type': 'error',
                        'code': f'# Error getting code: {e}',
                        'start_line': None,
                        'end_line': None,
                        'code_omitted': False,
                        'original_code_length': 0
                    })

            # 构建包含代码的链信息
            chain_with_code = {
                'chain_id': chain_id,
                'original_chain_info': chain_info,
                'entities_with_code': entities_with_code,
                'chain_length': len(entities_with_code),
                'selection_rank': chain_info.get('selection_rank', i + 1),
                'is_longest': chain_info.get('is_longest', False)
            }

            chains_with_code.append(chain_with_code)

            # 统计代码省略情况
            omitted_count = sum(1 for entity in entities_with_code if entity.get('code_omitted', False))
            total_code_length = sum(entity.get('original_code_length', 0) for entity in entities_with_code)

            logging.info(f"链 {chain_id} 处理完成，包含 {len(entities_with_code)} 个实体代码")
            logging.info(f"  其中 {omitted_count} 个实体代码被省略，总代码长度: {total_code_length}")

        logging.info(f"阶段5完成，所有 {len(chains_with_code)} 条链都已添加代码信息")
        return chains_with_code

    def _vote_on_chains(self, chains_with_code: List[Dict[str, Any]], issue_description: str, num_agents: int = 5) -> \
            Dict[str, Any]:
        """
        Stage 6: 使用多个agent对定位链进行投票

        Args:
            chains_with_code: 包含代码的定位链列表
            issue_description: 问题描述
            num_agents: 投票agent数量

        Returns:
            投票结果
        """
        logging.info(f"=== Stage 6: 使用 {num_agents} 个agent对 {len(chains_with_code)} 条定位链进行投票 ===")

        if not chains_with_code:
            logging.warning("没有可投票的定位链")
            return {
                'success': False,
                'error': 'No chains to vote on',
                'votes': [],
                'winning_chain': None
            }

        # 构建chains信息字符串
        chains_info = self._format_chains_for_voting(chains_with_code)

        # 使用多线程并行进行投票
        vote_results = []
        vote_lock = threading.Lock()

        def vote_worker(agent_id: int) -> Dict[str, Any]:
            """单个agent的投票工作函数"""
            try:
                logging.info(f"Agent {agent_id} 开始投票...")

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
                    temp=0.7,  # 适当的温度保证多样性
                    max_tokens=2000
                )

                # 解析投票结果
                vote_result = self._parse_vote_result(response, agent_id)

                with vote_lock:
                    vote_results.append(vote_result)
                    logging.info(f"Agent {agent_id} 投票完成: {vote_result.get('voted_chain_id', 'unknown')}")

                return vote_result

            except Exception as e:
                logging.error(f"Agent {agent_id} 投票失败: {e}")
                return {
                    'agent_id': agent_id,
                    'voted_chain_id': None,
                    'confidence': 0,
                    'reasoning': f'投票失败: {e}',
                    'error': str(e)
                }

        # 并行执行投票
        with ThreadPoolExecutor(max_workers=min(num_agents, 3)) as executor:
            futures = [executor.submit(vote_worker, i + 1) for i in range(num_agents)]

            for future in as_completed(futures):
                try:
                    future.result()  # 等待完成，错误已在worker中处理
                except Exception as e:
                    logging.error(f"投票线程异常: {e}")

        # 统计投票结果
        voting_summary = self._analyze_voting_results(vote_results, chains_with_code)

        logging.info(f"投票完成，获胜链: {voting_summary.get('winning_chain_id', 'None')}")
        return voting_summary

    def _format_chains_for_voting(self, chains_with_code: List[Dict[str, Any]]) -> str:
        """格式化定位链信息用于投票（简化版本，只包含实体和完整代码）"""
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
        """解析单个agent的投票结果"""
        try:
            # 清理响应文本
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            vote_data = json.loads(response_text)
            vote_data['agent_id'] = agent_id

            # 验证必要字段
            if 'voted_chain_id' not in vote_data:
                raise ValueError("Missing voted_chain_id")

            logging.info(f"Agent {agent_id} 投票解析成功: {vote_data['voted_chain_id']}")
            return vote_data

        except Exception as e:
            logging.error(f"Agent {agent_id} 投票结果解析失败: {e}")
            return {
                'agent_id': agent_id,
                'voted_chain_id': None,
                'confidence': 0,
                'reasoning': f'解析失败: {e}',
                'error': str(e)
            }

    def _analyze_voting_results(self, vote_results: List[Dict[str, Any]], chains_with_code: List[Dict[str, Any]]) -> \
            Dict[str, Any]:
        """分析投票结果"""
        logging.info("分析投票结果...")

        # 统计有效投票
        valid_votes = [v for v in vote_results if v.get('voted_chain_id') and 'error' not in v]
        invalid_votes = [v for v in vote_results if 'error' in v or not v.get('voted_chain_id')]

        logging.info(f"有效投票: {len(valid_votes)}, 无效投票: {len(invalid_votes)}")

        if not valid_votes:
            logging.warning("没有有效投票")
            return {
                'success': False,
                'error': 'No valid votes received',
                'votes': vote_results,
                'winning_chain': None
            }

        # 统计每条链的得票数
        vote_counts = Counter(v['voted_chain_id'] for v in valid_votes)

        # 找到获胜链
        winning_chain_id = vote_counts.most_common(1)[0][0]
        winning_votes = vote_counts[winning_chain_id]

        # 找到对应的链数据
        winning_chain_data = None
        for chain in chains_with_code:
            if chain['chain_id'] == winning_chain_id:
                winning_chain_data = chain
                break

        # 计算平均置信度
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

        logging.info(f"投票结果分析完成:")
        logging.info(f"  获胜链: {winning_chain_id}")
        logging.info(f"  得票数: {winning_votes}/{len(valid_votes)}")
        logging.info(f"  平均置信度: {avg_confidence:.1f}")
        logging.info(f"  投票分布: {dict(vote_counts)}")

        return voting_summary

    def _generate_modification_plan(self, winning_chain: Dict[str, Any], issue_description: str, num_agents: int = 5, 
                                   instance_id: str = None, cache_timestamp: str = None) -> \
            Dict[str, Any]:
        """
        Stage 7: 生成修改plan的多轮agent讨论

        Args:
            winning_chain: 获胜的定位链
            issue_description: 问题描述
            num_agents: 参与讨论的agent数量
            instance_id: 实例ID，用于缓存
            cache_timestamp: 缓存时间戳

        Returns:
            最终的修改plan
        """
        logging.info(f"=== Stage 7: 使用 {num_agents} 个agent进行修改plan讨论 ===")

        # 准备chain信息
        chain_info = self._format_chain_for_modification_discussion(winning_chain)

        # 第一轮：每个agent独立分析修改位置
        logging.info("第一轮: 每个agent独立分析修改位置")
        first_round_analyses = self._conduct_first_round_analysis(
            chain_info, issue_description, num_agents, instance_id, cache_timestamp
        )
        
        # 保存第一轮分析结果
        if instance_id and cache_timestamp:
            stage7_round1_data = {
                'first_round_analyses': first_round_analyses,
                'chain_info': chain_info,
                'num_agents': num_agents
            }
            self._save_stage_result(instance_id, 'stage_7_round1_analysis', stage7_round1_data, cache_timestamp)

        # 第二轮：每个agent基于其他agent的分析进行综合判断
        logging.info("第二轮: agent综合其他agent的分析进行判断")
        second_round_analyses = self._conduct_second_round_analysis(
            chain_info, issue_description, first_round_analyses, instance_id, cache_timestamp
        )
        
        # 保存第二轮分析结果
        if instance_id and cache_timestamp:
            stage7_round2_data = {
                'second_round_analyses': second_round_analyses,
                'first_round_analyses': first_round_analyses
            }
            self._save_stage_result(instance_id, 'stage_7_round2_analysis', stage7_round2_data, cache_timestamp)

        # 第三轮：discriminator进行最终判定
        logging.info("第三轮: discriminator进行最终判定")
        final_plan = self._conduct_final_discrimination(
            chain_info, issue_description, second_round_analyses, instance_id, cache_timestamp
        )
        
        # 保存最终判定结果
        if instance_id and cache_timestamp:
            stage7_final_data = {
                'final_plan': final_plan,
                'second_round_analyses': second_round_analyses
            }
            self._save_stage_result(instance_id, 'stage_7_final_plan', stage7_final_data, cache_timestamp)

        logging.info(
            f"修改plan生成完成，包含 {len(final_plan.get('final_plan', {}).get('modifications', []))} 个修改步骤")
        return final_plan

    def _format_chain_for_modification_discussion(self, winning_chain: Dict[str, Any]) -> str:
        """格式化获胜链信息用于修改讨论"""
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
        """第一轮分析：每个agent独立分析修改位置"""
        first_round_results = []

        def analyze_worker(agent_id: int) -> Dict[str, Any]:
            try:
                logging.info(f"Agent {agent_id} 开始第一轮分析...")

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
                logging.info(f"Agent {agent_id} 第一轮分析完成")
                return analysis

            except Exception as e:
                logging.error(f"Agent {agent_id} 第一轮分析失败: {e}")
                return {
                    'agent_id': agent_id,
                    'round': 'first_round',
                    'analysis': None,
                    'error': str(e)
                }

        # 并行执行第一轮分析
        with ThreadPoolExecutor(max_workers=min(num_agents, 3)) as executor:
            futures = [executor.submit(analyze_worker, i + 1) for i in range(num_agents)]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    first_round_results.append(result)
                except Exception as e:
                    logging.error(f"第一轮分析线程异常: {e}")

        valid_analyses = [r for r in first_round_results if r.get('analysis')]
        logging.info(f"第一轮分析完成，有效分析: {len(valid_analyses)}/{num_agents}")

        return first_round_results
    
    def _conduct_second_round_analysis(self, chain_info: str, issue_description: str,
                                       first_round_analyses: List[Dict[str, Any]],
                                       instance_id: str = None, cache_timestamp: str = None) -> List[Dict[str, Any]]:
        """第二轮分析：agent综合其他agent的分析"""
        second_round_results = []

        # 准备其他agent的分析摘要
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
                    for j, mod in enumerate(modifications):  # 限制显示前3个[:3]，还是不限制好
                        summary += f"  {j + 1}. {mod.get('entity_id', 'N/A')}: {mod.get('location_description', 'N/A')}\n"
                        summary += f"     Priority: {mod.get('priority', 'N/A')}, Type: {mod.get('modification_type', 'N/A')}\n"
                summary += "\n"

            return summary if summary else "No other valid analyses available."

        def analyze_worker_round2(agent_analysis: Dict[str, Any]) -> Dict[str, Any]:
            agent_id = agent_analysis.get('agent_id')

            try:
                logging.info(f"Agent {agent_id} 开始第二轮分析...")

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
                logging.info(f"Agent {agent_id} 第二轮分析完成")
                return refined_analysis

            except Exception as e:
                logging.error(f"Agent {agent_id} 第二轮分析失败: {e}")
                return {
                    'agent_id': agent_id,
                    'round': 'second_round',
                    'analysis': None,
                    'error': str(e)
                }

        # 只对第一轮有效分析的agent进行第二轮
        valid_first_round = [a for a in first_round_analyses if a.get('analysis')]

        # 并行执行第二轮分析
        with ThreadPoolExecutor(max_workers=min(len(valid_first_round), 1)) as executor:
            futures = [executor.submit(analyze_worker_round2, analysis) for analysis in valid_first_round]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    second_round_results.append(result)
                except Exception as e:
                    logging.error(f"第二轮分析线程异常: {e}")

        valid_second_round = [r for r in second_round_results if r.get('analysis')]
        logging.info(f"第二轮分析完成，有效分析: {len(valid_second_round)}/{len(valid_first_round)}")

        return second_round_results

    def _conduct_final_discrimination(self, chain_info: str, issue_description: str,
                                      second_round_analyses: List[Dict[str, Any]], 
                                      instance_id: str = None, cache_timestamp: str = None) -> Dict[str, Any]:
        """第三轮：discriminator进行最终判定"""
        logging.info("Discriminator开始最终判定...")

        # 准备所有agent的最终分析
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
                    all_agents_summary += f"     Reasoning: {mod.get('reasoning', 'N/A')[:200]}...\n"
            all_agents_summary += "\n"

        # 使用discriminator进行最终判定
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
                temp=0.3,  # 降低温度以获得更稳定的判断
                max_tokens=5000
            )

            final_plan = self._parse_final_plan(response)
            logging.info("Discriminator最终判定完成")
            return final_plan

        except Exception as e:
            logging.error(f"Discriminator最终判定失败: {e}")
            return {
                'success': False,
                'error': f'Final discrimination failed: {e}',
                'final_plan': {
                    'summary': 'Failed to generate plan',
                    'modifications': []
                }
            }

    def _parse_modification_analysis(self, response: str, agent_id: int, round_name: str) -> Dict[str, Any]:
        """解析修改分析结果"""
        try:
            # 清理响应文本
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
            logging.error(f"Agent {agent_id} {round_name} 分析结果解析失败: {e}")
            return {
                'agent_id': agent_id,
                'round': round_name,
                'analysis': None,
                'error': str(e)
            }

    def _parse_final_plan(self, response: str) -> Dict[str, Any]:
        """解析最终的修改plan"""
        try:
            # 清理响应文本
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            plan_data = json.loads(response_text)

            # 验证plan格式
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
            logging.error(f"最终plan解析失败: {e}")
            return {
                'success': False,
                'error': f'Final plan parsing failed: {e}',
                'final_plan': {
                    'summary': 'Failed to parse plan',
                    'modifications': []
                }
            }

    def _extract_entity_groups(self, initial_entities: List[str], problem_statement: str) -> List[Dict[str, Any]]:
        """提取entity组的辅助方法"""
        entity_groups = []
        for i, initial_entity in enumerate(initial_entities):
            logging.info(f"处理初始entity {i + 1}/{len(initial_entities)}: '{initial_entity}'")
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
        """生成所有定位链的辅助方法"""
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

            # 收集所有有效的定位链
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
        格式化输出给edit agent的信息

        Args:
            issue_description: 问题描述
            modification_plan: 修改计划
            winning_chain: 获胜的定位链

        Returns:
            格式化的prompt字符串
        """
        logging.info("开始格式化edit agent prompt")

        # 第一部分：Issue
        issue_section = f"<issue>\n{issue_description}\n</issue>\n\n"

        # 第二部分：Plan（使用醒目的stage标记）
        plan_section = "<plan>\n"

        final_plan = modification_plan.get('final_plan', {})
        modifications = final_plan.get('modifications', [])

        for i, modification in enumerate(modifications):
            step = modification.get('step', i + 1)
            instruction = modification.get('instruction', 'No instruction provided')
            context = modification.get('context', 'No context provided')

            # 使用醒目的stage标记格式
            plan_section += f"***stage {step}***\n"
            plan_section += f"instruction: {instruction}\n"
            plan_section += f"context: {context}\n\n"

        plan_section += "</plan>\n\n"

        # 第三部分：代码片段（处理已包含行号的代码）
        code_section = "<code>\n"

        entities_with_code = winning_chain.get('entities_with_code', [])

        # 按文件路径分组实体
        file_entities = {}
        for entity in entities_with_code:
            entity_id = entity.get('entity_id', '')
            code = entity.get('code', '')

            # 提取文件路径
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

        # 为每个文件格式化代码
        for file_path, entities in file_entities.items():
            code_section += f"{file_path}:\n"

            for entity in entities:
                code = entity['code']

                if code and code.strip():
                    # 清理代码格式（移除markdown代码块标记但保留行号）
                    formatted_code = self._clean_code_format(code)
                    code_section += formatted_code + "\n"
                else:
                    code_section += f"# No code available for {entity['entity_id']}\n"

            code_section += "\n"  # 文件之间的分隔

        code_section += "</code>"

        # 组合所有部分
        full_prompt = issue_section + plan_section + code_section

        logging.info(f"edit agent prompt格式化完成，总长度: {len(full_prompt)}")
        logging.info(f"包含 {len(file_entities)} 个文件的代码")

        return full_prompt

    def _clean_code_format(self, code: str) -> str:
        """
        清理代码格式，移除markdown标记但保留原有的行号

        Args:
            code: 原始代码字符串，可能包含markdown格式和行号

        Returns:
            清理后的代码字符串
        """
        if not code or not code.strip():
            return "# No code content"

        # 移除开头和结尾的markdown代码块标记
        cleaned_code = code.strip()

        # 移除开头的```或```python等
        if cleaned_code.startswith('```'):
            lines = cleaned_code.split('\n')
            # 找到第一行```之后的内容
            start_idx = 1
            if len(lines) > 1:
                cleaned_code = '\n'.join(lines[start_idx:])

        # 移除结尾的```
        if cleaned_code.endswith('```'):
            lines = cleaned_code.split('\n')
            # 移除最后的```行
            if lines and lines[-1].strip() == '```':
                cleaned_code = '\n'.join(lines[:-1])

        # 处理行号格式：如果代码已经包含行号（格式如 "102 | def page_range"），直接返回
        lines = cleaned_code.split('\n')
        formatted_lines = []

        for line in lines:
            # 检查是否已经有行号格式（数字 + | 或数字 + 空格）
            stripped_line = line.strip()
            if stripped_line and (
                    # 格式1: "102 | def page_range"
                    ('|' in line and line.split('|')[0].strip().isdigit()) or
                    # 格式2: "102    def page_range" (数字开头且后面有空格)
                    (len(line) > 0 and line.split()[0].isdigit() and len(line.split()) > 1)
            ):
                # 已经有行号，保持原格式但统一为空格分隔
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
                # 没有行号的行（可能是空行或注释），直接保留
                formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def _add_line_numbers_to_code(self, code: str, start_line: Optional[int] = None) -> str:
        """
        为代码添加行号（已废弃，被_clean_code_format替代）

        Args:
            code: 源代码
            start_line: 起始行号，如果为None则从1开始

        Returns:
            带行号的代码
        """
        # 这个方法已被_clean_code_format替代，但保留以避免破坏其他可能的调用
        return self._clean_code_format(code)

    def _ensure_cache_dir_exists(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_file_path(self, instance_id: str, timestamp: str = None) -> str:
        """获取缓存文件路径"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        instance_dir = os.path.join(self.cache_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        
        cache_file = os.path.join(instance_dir, f"{timestamp}_pipeline_cache.json")
        return cache_file
    
    def _save_stage_result(self, instance_id: str, stage_name: str, stage_data: Dict[str, Any], 
                          timestamp: str = None):
        """保存单个stage的结果"""
        if not self.enable_cache:
            return
            
        try:
            cache_file = self._get_cache_file_path(instance_id, timestamp)
            
            # 读取现有缓存或创建新的
            cache_data = {}
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # 添加stage结果
            cache_data[stage_name] = {
                'timestamp': datetime.now().isoformat(),
                'data': stage_data
            }
            
            # 保存到文件
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"已保存 {stage_name} 结果到缓存文件: {cache_file}")
            
        except Exception as e:
            logging.error(f"保存stage结果失败: {e}")
    
    def _load_stage_result(self, instance_id: str, stage_name: str, timestamp: str = None) -> Optional[Dict[str, Any]]:
        """加载单个stage的结果"""
        if not self.enable_cache:
            return None
            
        try:
            cache_file = self._get_cache_file_path(instance_id, timestamp)
            
            if not os.path.exists(cache_file):
                return None
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if stage_name in cache_data:
                logging.info(f"从缓存加载 {stage_name} 结果: {cache_file}")
                return cache_data[stage_name]['data']
            
            return None
            
        except Exception as e:
            logging.error(f"加载stage结果失败: {e}")
            return None
    
    def _list_cached_instances(self) -> List[str]:
        """列出所有缓存的instance"""
        try:
            if not os.path.exists(self.cache_dir):
                return []
            return [d for d in os.listdir(self.cache_dir) 
                   if os.path.isdir(os.path.join(self.cache_dir, d))]
        except Exception as e:
            logging.error(f"列出缓存实例失败: {e}")
            return []
    
    def _list_cached_files_for_instance(self, instance_id: str) -> List[str]:
        """列出指定instance的所有缓存文件"""
        try:
            instance_dir = os.path.join(self.cache_dir, instance_id)
            if not os.path.exists(instance_dir):
                return []
            return [f for f in os.listdir(instance_dir) if f.endswith('_pipeline_cache.json')]
        except Exception as e:
            logging.error(f"列出实例缓存文件失败: {e}")
            return []

