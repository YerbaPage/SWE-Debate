import json
import logging
import os
from enum import Enum
from textwrap import dedent
from typing import Optional, Union, List, Any

import litellm
import tenacity
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
)
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.completion.model import Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """估算文本的token数量 (简单估算: 1 token ≈ 4个字符)"""
    if not text:
        return 0
    return len(text) // 4


def truncate_messages_if_needed(messages: list[dict], max_tokens: int = 60000) -> list[dict]:
    """如果消息超过token限制，则截断消息列表
    
    Args:
        messages: 消息列表
        max_tokens: 最大token数量限制
        
    Returns:
        截断后的消息列表
    """
    if not messages:
        return messages
        
    # 计算总token数
    total_tokens = sum(
        estimate_tokens(msg.get("content", ""))
        for msg in messages
    )
    
    if total_tokens <= max_tokens:
        return messages
        
    logger.warning(f"消息总长度 {total_tokens} tokens 超过限制 {max_tokens} tokens，进行截断")
    
    # 保留系统消息和最后几条消息
    result_messages = []
    remaining_tokens = max_tokens
    
    # 首先保留系统消息
    system_messages = [msg for msg in messages if msg.get("role") == "system"]
    for msg in system_messages:
        tokens = estimate_tokens(msg.get("content", ""))
        if tokens <= remaining_tokens:
            result_messages.append(msg)
            remaining_tokens -= tokens
        else:
            # 截断系统消息
            content = msg.get("content", "")
            max_chars = remaining_tokens * 4
            truncated_content = content[:max_chars] + "...[截断]"
            result_messages.append({
                "role": msg.get("role"),
                "content": truncated_content
            })
            remaining_tokens = 0
            break
    
    # 然后从后往前添加非系统消息
    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
    for msg in reversed(non_system_messages):
        tokens = estimate_tokens(msg.get("content", ""))
        if tokens <= remaining_tokens:
            result_messages.insert(-len(system_messages) if system_messages else 0, msg)
            remaining_tokens -= tokens
        else:
            # 如果还有剩余token，截断这条消息
            if remaining_tokens > 100:  # 至少保留100个token
                content = msg.get("content", "")
                max_chars = remaining_tokens * 4
                truncated_content = content[:max_chars] + "...[截断]"
                result_messages.insert(-len(system_messages) if system_messages else 0, {
                    "role": msg.get("role"),
                    "content": truncated_content
                })
            break
    
    final_tokens = sum(estimate_tokens(msg.get("content", "")) for msg in result_messages)
    logger.info(f"消息截断完成: {total_tokens} -> {final_tokens} tokens")
    
    return result_messages


class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


class CompletionResponse(BaseModel):
    """Container for completion responses that can include multiple structured outputs and text"""

    structured_outputs: List[StructuredOutput] = Field(default_factory=list)
    text_response: Optional[str] = Field(default=None)
    completion: Optional[Completion] = Field(default=None)

    @classmethod
    def create(
        cls,
        text: str | None = None,
        output: List[StructuredOutput] | StructuredOutput | None = None,
        completion: Completion | None = None,
    ) -> "CompletionResponse":
        if isinstance(output, StructuredOutput):
            outputs = [output]
        elif isinstance(output, list):
            outputs = output
        else:
            outputs = None

        return cls(
            text_response=text, structured_outputs=outputs, completion=completion
        )

    @property
    def structured_output(self) -> Optional[StructuredOutput]:
        """Get the first structured output"""
        if len(self.structured_outputs) > 1:
            ignored_outputs = [
                output.__class__.__name__ for output in self.structured_outputs[1:]
            ]
            logger.warning(
                f"Multiple structured outputs found in completion response, returning {self.structured_outputs[0].__class__.__name__} and ignoring: {ignored_outputs}"
            )
        return self.structured_outputs[0] if self.structured_outputs else None


class CompletionModel(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    temperature: float = Field(0.0, description="The temperature to use for completion")
    max_tokens: int = Field(
        2000, description="The maximum number of tokens to generate"
    )
    timeout: float = Field(
        120.0, description="The timeout in seconds for completion requests"
    )
    model_base_url: Optional[str] = Field(
        default=None, description="The base URL for the model API"
    )
    model_api_key: Optional[str] = Field(
        default=None, description="The API key for the model", exclude=True
    )
    response_format: Optional[LLMResponseFormat] = Field(
        None, description="The response format expected from the LLM"
    )
    stop_words: Optional[list[str]] = Field(
        default=None, description="The stop words to use for completion"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Additional metadata for the completion model"
    )
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    def clone(self, **kwargs) -> "CompletionModel":
        """Create a copy of the completion model with optional parameter overrides.

        Args:
            **kwargs: Parameters to override in the cloned model

        Returns:
            A new CompletionModel instance with the specified overrides
        """
        model_data = self.model_dump()
        model_data.update(kwargs)
        return CompletionModel.model_validate(model_data)

    def create_completion(
        self,
        messages: List[dict],
        system_prompt: str,
        response_model: List[type[StructuredOutput]] | type[StructuredOutput],
    ) -> CompletionResponse:
        if not response_model:
            raise CompletionRuntimeError(f"Response model is required for completion")

        if isinstance(response_model, list) and len(response_model) > 1:
            avalabile_actions = [
                action for action in response_model if hasattr(action, "name")
            ]
            if not avalabile_actions:
                raise CompletionRuntimeError(f"No actions found in {response_model}")

            class TakeAction(StructuredOutput):
                action: Union[tuple(response_model)] = Field(...)
                action_type: str = Field(
                    ..., description="The type of action being taken"
                )

                @model_validator(mode="before")
                def validate_action(cls, data: dict) -> dict:
                    if not isinstance(data, dict):
                        raise ValidationError("Expected dictionary input")

                    action_type = data.get("action_type")
                    if not action_type:
                        return data

                    # Find the correct action class based on action_type
                    action_class = next(
                        (
                            action
                            for action in avalabile_actions
                            if action.name == action_type
                        ),
                        None,
                    )
                    if not action_class:
                        action_names = [action.name for action in avalabile_actions]
                        raise ValidationError(
                            f"Unknown action type: {action_type}. Available actions: {', '.join(action_names)}"
                        )

                    # Validate the action data using the specific action class
                    action_data = data.get("action")
                    if not action_data:
                        raise ValidationError("Action data is required")

                    data["action"] = action_class.model_validate(action_data)
                    return data

            response_model = TakeAction

        system_prompt += dedent(f"""\n# Response format
        You must respond with only a JSON object that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself.""")

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError, TypeError)
            ),
            stop=tenacity.stop_after_attempt(4),
            wait=tenacity.wait_exponential(multiplier=4),
        )

        def _do_completion():
            completion_response = None
    
            try:
                # completion_response = self._litellm_base_completion(
                #     messages=messages, response_format={"type": "json_object"}
                # )
                completion_response = self._litellm_base_completion(
                    messages=messages
                )
        
                # print(f'completion_response {completion_response}')
                if not completion_response or not completion_response.choices:
                    raise CompletionRuntimeError(
                        "No completion response or choices returned"
                    )
        
                if isinstance(
                    completion_response.choices[0].message.content, BaseModel
                ):
                    assistant_message = completion_response.choices[
                        0
                    ].message.content.model_dump()
                else:
                    assistant_message = completion_response.choices[0].message.content
        
                if not assistant_message:
                    raise CompletionRuntimeError("Empty response from model")
        
                messages.append({"role": "assistant", "content": assistant_message})
        
                response = response_model.model_validate_json(assistant_message)
        
                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=completion_response,
                    model=self.model,
                )
                if hasattr(response, "action"):
                    return CompletionResponse.create(
                        output=response.action, completion=completion
                    )
        
                return CompletionResponse.create(output=response, completion=completion)

            except (ValidationError, json.JSONDecodeError) as e:
                logger.warning(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"The response was invalid. Fix the errors, exceptions found\n{e}",
                    }
                )
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                ) from e
            except Exception as e:
                logger.exception(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                raise CompletionRuntimeError(
                    f"Failed to get completion response: {e}",
                )

        # _do_completion
        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _litellm_base_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
    ) -> Any:
        """Base method for making litellm completion calls with common parameters.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions for function calling
            tool_choice: Optional tool choice configuration
            response_format: Optional response format configuration

        Returns:
            The completion response from litellm
        """
        litellm.drop_params = True

        # 检查并截断消息以适应模型的上下文窗口
        if "deepseek" in self.model.lower():
            # DeepSeek模型的上下文窗口是65536 tokens，保留一些余量
            max_input_tokens = 60000  # 保留5536 tokens用于生成响应
            messages = truncate_messages_if_needed(messages, max_input_tokens)
        elif "gpt-4" in self.model.lower():
            # GPT-4模型的上下文窗口是8192 tokens
            max_input_tokens = 6000  # 保留2192 tokens用于生成响应
            messages = truncate_messages_if_needed(messages, max_input_tokens)
        else:
            # 其他模型使用默认限制
            max_input_tokens = 60000
            messages = truncate_messages_if_needed(messages, max_input_tokens)

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(6),
            wait=tenacity.wait_exponential(multiplier=3),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying litellm completion after error: {retry_state.outcome.exception()}"
            ),
        )
        def _do_completion():
            nonlocal messages
            try:
                return litellm.completion(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                    metadata=self.metadata or {},
                    timeout=self.timeout,
                    api_base=self.model_base_url,
                    api_key=self.model_api_key,
                    stop=self.stop_words,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    request_timeout=self.timeout,
                )
            except Exception as e:
                error_msg = str(e)
                if "ContextWindowExceededError" in error_msg or "context length" in error_msg.lower():
                    # 如果仍然超过上下文窗口，进行更激进的截断
                    logger.warning(f"上下文窗口仍然超限，进行更激进的截断: {error_msg}")
                    if "deepseek" in self.model.lower():
                        messages = truncate_messages_if_needed(messages, 45000)  # 更保守的限制
                    else:
                        messages = truncate_messages_if_needed(messages, 4000)   # 更保守的限制
                raise e

        try:
            return _do_completion()
        except tenacity.RetryError as e:
            last_exception = e.last_attempt.exception()
            if isinstance(last_exception, litellm.APIError):
                logger.error(
                    "LiteLLM API Error: %s\nProvider: %s\nModel: %s\nStatus: %d\nDebug Info: %s\nRetries: %d/%d",
                    last_exception.message,
                    last_exception.llm_provider,
                    last_exception.model,
                    last_exception.status_code,
                    last_exception.litellm_debug_info,
                    last_exception.num_retries or 0,
                    last_exception.max_retries or 0,
                )
            else:
                logger.warning(
                    "LiteLLM completion failed after retries with error: %s",
                    str(last_exception),
                    exc_info=last_exception,
                )
            raise last_exception

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "model_api_key" in dump:
            dump["model_api_key"] = None
        if "response_format" in dump:
            dump["response_format"] = dump["response_format"].value
        return dump

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "response_format" in obj:
            if "claude-3-5" in obj["model"]:
                from moatless.completion.anthropic import AnthtropicCompletionModel

                return AnthtropicCompletionModel(**obj)

            response_format = LLMResponseFormat(obj["response_format"])
            obj["response_format"] = response_format

            if response_format == LLMResponseFormat.TOOLS:
                from moatless.completion.tool_call import ToolCallCompletionModel

                return ToolCallCompletionModel(**obj)
            elif response_format == LLMResponseFormat.REACT:
                from moatless.completion.react import ReActCompletionModel

                return ReActCompletionModel(**obj)

        return cls(**obj)

    @model_validator(mode="after")
    def set_api_key(self) -> "CompletionModel":
        """
        Update the model with the API key from en vars if model base URL is set but API key is not as we don't persist the API key.
        """
        if self.model_base_url and not self.model_api_key:
            self.model_api_key = os.getenv("CUSTOM_LLM_API_KEY")

        return self
