import json
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.completion.model import Completion


class Instructor(BaseModel):
    system_prompt: str = Field(
        ..., description="System prompt to be used for generating completions"
    )
    _completion: CompletionModel = PrivateAttr()
    
    
    def __init__(
        self,
        completion: CompletionModel,
        system_prompt: str | None = None,
        **data,
    ):
        super().__init__(
            system_prompt=system_prompt,
            **data,
        )
        self._completion = completion

    
    def instruct(self, messages):
        # messages = self.message_generator.generate(node)
        messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        response = self._completion._litellm_base_completion(
                    messages=messages
                ).choices[0].message.content
        response = json.loads(response)
        thoughts, instructions, ty = response['thoughts'], response['instructions'], response['type']

        return thoughts, instructions, ty