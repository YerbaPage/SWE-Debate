import os
import json
import random
from typing import Optional, Dict, Any, List, Callable, Union
from pydantic import BaseModel, Field, model_validator
from textwrap import dedent
from litellm.types.llms.openai import ChatCompletionUserMessage

from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.model import StructuredOutput, Completion
from moatless.completion.completion import CompletionModel, CompletionResponse

from moatless.repository.repository import Repository
from moatless.benchmark.swebench import create_repository
from moatless.index import CodeIndex
from moatless.file_context import FileContext
from moatless.selector import BestFirstSelector, Selector, SoftmaxSelector, LLMSelector
from moatless.selector.feedback_selector import FeedbackSelector
from moatless.value_function.base import ValueFunction

from moatless.actions.action import Action
from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, Reject, RunTests, StringReplace, CreateFile
from moatless.agent.code_agent import CodingAgent, create_edit_code_actions
from moatless.agent.code_prompts import *
from moatless.agent.agent import ActionAgent
from moatless.search_tree import SearchTree
from moatless.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)
from moatless.schema import MessageHistoryType
from moatless.message_history import MessageHistoryGenerator
from moatless.agent.settings import AgentSettings
from moatless.node import Node, ActionStep
from moatless.expander import Expander
from moatless.value_function.model import Reward
from moatless.exceptions import RuntimeError, RejectError
persist_path = "/root/autodl-tmp/moatless-tree-search-main/tmp/trajectory/django__django-10914-trajectory.json"



class SilinSearchTree(BaseModel):
    root: Node = Field(..., description="The root node of the search tree.")
    selector: Union[
        BestFirstSelector, SoftmaxSelector, LLMSelector, FeedbackSelector
    ] = Field(..., description="Selector for node selection.")
    agent: ActionAgent = Field(..., description="Agent for generating actions.")
    actions: List[Action] = Field(
        default_factory=list,
        description="Actions that can be used by the agent in the search tree.",
    )
    repository: Optional[Repository] = Field(
        None, description="Repository for the search tree."
    )
    expander: Optional[Expander] = Field(
        None, description="Expander for expanding nodes."
    )
    value_function: Optional[ValueFunction] = Field(
        None, description="Value function for reward calculation."
    )
    persist_path: Optional[str] = Field(
        None, description="Path to persist the search tree."
    )
    unique_id: int = Field(default=0, description="Unique ID counter for nodes.")

    max_expansions: int = Field(
        1, description="The maximum number of expansions of one state."
    )
    max_iterations: int = Field(
        10, description="The maximum number of iterations to run the tree search."
    )
    min_finished_nodes: Optional[int] = Field(
        None,
        description="The minimum number of finished nodes to consider before finishing",
    )
    max_finished_nodes: Optional[int] = Field(
        None,
        description="The maximum number of finished nodes to consider before finishing",
    )
    max_depth: Optional[int] = Field(
        None, description="The maximum depth for one trajectory in simulations."
    )
    
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(
        cls,
        message: Optional[str] = None,
        root: Optional[Node] = None,
        file_context: Optional[FileContext] = None,
        repository: Repository | None = None,
        expander: Expander | None = None,
        selector: Optional[Selector] = None,
        agent: Optional[ActionAgent] = None,
        value_function: Optional[ValueFunction] = None,
        persist_path: Optional[str] = None,
        max_expansions: int = 1,
        max_iterations: int = 10,
        max_depth: int = 10,
    ) -> "SilinSearchTree":
        if not root and not message:
            raise ValueError("Either a root node or a message must be provided.")

        if not file_context:
            file_context = FileContext(repo=repository)

        if not root:
            root = Node(
                node_id=0,
                max_expansions=max_expansions,
                user_message=message,
                file_context=file_context,
            )

        selector = selector or BestFirstSelector()
        expander = Expander(max_expansions=max_expansions)

        return cls(
            root=root,
            selector=selector,
            expander=expander,
            agent=agent,
            value_function=value_function,
            persist_path=persist_path,
            max_expansions=max_expansions,
            max_iterations=max_iterations,
            max_depth=max_depth,
        )

    def run_search(self) -> Node | None:
        """Run the MCTS algorithm for a specified number of iterations."""
        # if len(self.root.get_all_nodes()) > 1:
        #     self.log(
        #         logger.info,
        #         f"Restarting search tree with {len(self.root.get_all_nodes())} nodes",
        #     )

        while not self.is_finished():

            node = self._select(self.root)

            if node:
                new_node = self._expand(node)
                self._simulate(new_node)
                self._backpropagate(new_node)
                # self.maybe_persist()
            else:
                print("Search complete: no more nodes to expand.")
                break

        if not len(self.get_finished_nodes()):
            print(
                f"Search completed with no finished nodes. {len(self.root.get_all_nodes())} nodes created.",
            )
        else:
            print(
                f"Search completed with {len(self.get_finished_nodes())} finished nodes. {len(self.root.get_all_nodes())} nodes created.",
            )

        return self.get_all_trajectory()

    def _select(self, node: Node) -> Optional[Node]:
        """Select a node for expansion using the UCT algorithm."""
        expandable_nodes = node.get_expandable_descendants()

        # if not expandable_nodes:
        #     print("No expandable nodes found.")
        #     return None

        #         if expandable_nodes and self.finish_before_reexpanding:
        #     # Sort by node_id to get the most recently created node
        #     latest_node = max(expandable_nodes, key=lambda n: n.node_id)

        #     # Check if any node in the tree has reached a finished state
        #     all_nodes = node.get_all_nodes()
        #     has_finished_node = any(n.is_finished() for n in all_nodes)

        #     # Check if any node has exceeded the depth limit
        #     max_depth_exceeded = (
        #         any(
        #             n.get_depth() >= self.finish_before_reexpanding_depth
        #             for n in all_nodes
        #         )
        #         if self.finish_before_reexpanding_depth is not None
        #         else False
        #     )

        #     # Continue linear expansion only if no finished nodes exist and depth never exceeded
        #     if not has_finished_node and not max_depth_exceeded:
        #         return latest_node
        #     else:
        #         self.log(
        #             logger.info,
        #             f"Breaking linear path: {'finished state exists' if has_finished_node else 'depth limit exceeded'}",
        #         )

        # If we have a finished node or exceeded depth, use normal selection
        return self.selector.select(expandable_nodes)

    def _expand(self, node: Node, force_expansion: bool = False) -> Node:
        """Expand the node and return a child node."""

        # Check if any action step was not executed, if so return the node
        if node.action_steps and node.has_unexecuted_actions():
            print(
                f"Returning Node{node.node_id} with unexecuted actions"
            )
            return node

        child_node = self.expander.expand(node, self, force_expansion)

        if not node.action_steps and node.assistant_message:
            child_node.user_message = "You're an autonomous AI agent that must respond with one of the provided functions"

        # # Only add feedback if this is the second expansion from this node
        # if self.feedback_generator and len(node.children) >= 2:
        #     child_node.feedback_data = self.feedback_generator.generate_feedback(
        #         child_node,
        #         self.agent.actions,
        #     )

        print(
            f"Expanded Node{node.node_id} to new Node{child_node.node_id}"
        )
        return child_node

    def _simulate(self, node: Node):
        """Simulate a playout by executing the action and evaluating the result."""

        if node.observation:
            print(f"Node{node.node_id}: Action already executed. Skipping.")
        else:
            self.agent.run(node)

        if self.value_function and not node.is_duplicate and node.observation:
            try:
                node.reward, completion_response = self.value_function.get_reward(
                    node=node
                )
                node.completions["value_function"] = completion_response
                print(
                    f"Node{node.node_id}: The value function returned a reward of {node.reward.value}.",
                )
            except RejectError as e:
                print(
                    f"Node{node.node_id}: Value function rejected: {e.message}",
                )
                node.reward = None
            except RuntimeError as e:
                print(
                    f"Node{node.node_id}: Value function runtime error: {e.message}",
                )
                raise  # Re-raise to abort the entire search
    def _backpropagate(self, node: Node):
        """Backpropagate the reward up the tree."""
    
        if not node.reward:
            print(
                f"Node{node.node_id} has no evaluation. Skipping backpropagation.",
            )
            return
    
        reward = node.reward.value
        while node is not None:
            node.visits += 1
            if not node.value:
                node.value = reward
            else:
                node.value += reward
            node = node.parent

    def get_finished_nodes(self) -> List[Node]:
        """Get all finished nodes in the search tree by uniqe parent node."""
        parent_ids = set()
        finished_nodes = []
        for node in self.root.get_all_nodes():
            # TODO: Pick finished node with highest/avg/lowest reward?
            if node.is_finished() and node.parent.node_id not in parent_ids:
                parent_ids.add(node.parent.node_id)
                finished_nodes.append(node)

        return finished_nodes

    def is_finished(self):
        # Check max iterations
        if len(self.root.get_all_nodes()) >= self.max_iterations:
            print(
                f"Search finished: Reached max iterations {self.max_iterations}"
            )
            return True

        finished_nodes = self.get_finished_nodes()
        unique_finished_parents = set()
        for node in finished_nodes:
            unique_finished_parents.add(node.parent.node_id)

        # Check if there are no more expandable nodes
        expandable_nodes = self.root.get_expandable_descendants()
        if not expandable_nodes:
            print("Search finished: No more expandable nodes")
            return True

        return False

    
    def get_leaf_nodes(self) -> List[Node]:
        """Get all leaf nodes in the search tree."""
        return [node for node in self.root.get_all_nodes() if node.is_leaf()]

    def _generate_unique_id(self) -> int:
        self.unique_id += 1
        return self.unique_id
        
    def get_all_trajectory(self) -> Node | None:
        """
        Get all finished trajectory to return
        """

        nodes = self.get_finished_nodes()
        if not nodes:
            nodes = self.get_leaf_nodes()
            print(
                f"get_best_trajectory() No finished nodes found. Will select from {len(nodes)} leaf nodes.",
            )

        if len(nodes) == 1:
            return nodes[0]

        print(
                "No discriminator provided. Returning all the finished node.",
            )
        return nodes

        # if self.discriminator is None:
        #     self.log(
        #         logger.info,
        #         "No discriminator provided. Returning the first finished node.",
        #     )
        #     return nodes[-1]

        # return self.discriminator.select(nodes)



if __name__ == "__main__":
    # completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
    completion_model = CompletionModel(model="openai/moonshot-v1-128k", model_base_url=os.getenv("CUSTOM_LLM_API_BASE"), model_api_key=os.getenv("CUSTOM_LLM_API_KEY"), temperature=0.7)
    completion_model.response_format = LLMResponseFormat.TOOLS
    instance = get_moatless_instance(split='verified', instance_id="django__django-10914")  # 获得的instance是本地下载下来有点删改属性的swe-bench
    repository = create_repository(instance)
    code_index = CodeIndex.from_index_name(
        instance["instance_id"], file_repo=repository
    )
    file_context = FileContext(repo=repository)
    value_function = ValueFunction(completion_model=completion_model)
    actions = [
        FindClass(completion_model=completion_model, code_index=code_index, repository=repository),
        FindFunction(completion_model=completion_model, code_index=code_index, repository=repository),
        FindCodeSnippet(completion_model=completion_model, code_index=code_index, repository=repository),
        SemanticSearch(completion_model=completion_model, code_index=code_index, repository=repository),
        ViewCode(repository=repository),
        StringReplace(repository=repository, code_index=code_index),
        CreateFile(repository=repository, code_index=code_index),
        RunTests(repository=repository, code_index=code_index),
        Finish(),
        Reject()
    ]
    
    system_prompt = AGENT_ROLE
    if completion_model.response_format == LLMResponseFormat.REACT:
        system_prompt += REACT_CORE_OPERATION_RULES
    elif completion_model.response_format == LLMResponseFormat.TOOLS:
        system_prompt += REACT_GUIDELINES
    workflow_prompt = generate_workflow_prompt(actions, False)
    system_prompt += workflow_prompt + generate_guideline_prompt(False) + ADDITIONAL_NOTES
    
    agent = CodingAgent(system_prompt=system_prompt, actions=actions, completion=completion_model)
    # # 我认为应该是下面这种初始化，用的是内部的prompt而不是手动system_prompt，但是测试的时候是用了上面的初始化以及SIMPLE_CODE_PROMPT
    # agent = CodingAgent.create(repository=repository, completion_model=completion_model)
    # agent.actions = actions    # if not， 它内部的action没有code index，也没有repository
    
    search_tree = SilinSearchTree.create(
        message=instance["problem_statement"],
        agent=agent,
        file_context=file_context,
        value_function=value_function,
        max_iterations=100,
        max_expansions=3,
        max_depth=25,
        persist_path=persist_path,
    )

    node = search_tree._select(search_tree.root)
    new_node = search_tree._expand(node)
    search_tree._simulate(new_node)    