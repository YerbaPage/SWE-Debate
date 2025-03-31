# In the Stage 1,
# Action:{"action": {"thoughts": "To investigate the issue raised in the nested CompoundModels, we need to understand how the `separability_matrix` function computes separability for nested models. Therefore, the first step is to locate the implementation of the `separability_matrix` function within the astropy library.", "function_name": "separability_matrix"}, "action_type": "FindFunction"}
# Result: {
#   "description": "The last executed action was to find the 'separability_matrix' function within the astropy modeling module. This action is appropriate as it directly targets the function responsible for the computation of the separability matrix, which is central to the issue described. The output provides the definition of the function, which is exactly what's needed to diagnose the problem with the nested CompoundModels and their separability matrix computation. The function's code is found in the expected file 'astropy/modeling/separable.py', indicating the agent is on the right track. The code snippet is also of an appropriate size, providing enough context to understand the function's behavior without being overwhelming.",
#   "feedback": "An alternative branch could involve exploring how the 'CompoundModel' class handles nested models. This would provide insights into whether the issue is with the 'separability_matrix' function's logic or with how compound models are structured and processed. Specifically, investigate the '__and__' method within the 'CompoundModel' class to see how it constructs the compound model and whether this construction affects the separability matrix computation.",
#   "value": 95
# }

# In the Stage 2,
# Action:{"action": {"thoughts": "To further analyze the separability_matrix function and how it computes the correlation matrix, I need to view the implementation of _separable function which is the core of the separability_matrix function.", "files": [{"file_path": "astropy/modeling/separable.py", "start_line": 66, "end_line": 102, "span_ids": ["_separable"]}]}, "action_type": "ViewCode"}
# Result: {
#   "description": "The last executed action was to view the specific sections of code in the 'separable.py' file where the `separability_matrix` and `_separable` functions are defined. Given the problem statement, which describes an issue with the computation of the separability matrix for nested CompoundModels, examining these specific functions is highly relevant and appropriate. The `separability_matrix` function is the public interface for obtaining the separability matrix, and the `_separable` function seems to be a helper function that computes the separability logic for a given model. Reviewing these functions will likely provide insights into the logic that generates the separability matrix and help identify any potential bugs or misunderstandings in how nested CompoundModels are being processed.",
#   "feedback": "An alternative approach could involve running specific unit tests or creating test cases that exercise the code paths related to nested CompoundModels and their separability. This would help validate whether the observed behavior is indeed a bug or a correct implementation. Additionally, you could look for any known issues or discussions in the project's issue tracker related to the separability matrix, as this might provide more context or indicate if this is a known problem that has been previously reported or discussed by other developers.",
#   "value": 100
# }

# In the Stage 3,
# Action:{"action": {"tool": "FindCodeSnippet", "thoughts": "The issue is with the nested CompoundModels' separability_matrix computation. I need to investigate the _operators dictionary to understand how the operators are being handled within the compound models.", "file_pattern": null, "code_snippet": "_operators ="}, "action_type": "FindCodeSnippet"}
# Result: {
#   "description": "The executed action was to find the code snippet defining the '_operators' dictionary, which is a vital component related to the problem of nested CompoundModels in Astropy's separability matrix computation. The '_operators' dictionary maps modeling operators to functions that compute the relationship of axes as an array of 0s and 1s. This dictionary directly influences how the separability matrix calculates the correlation between outputs and inputs for compound models.",
#   "feedback": "Considering the '_operators' dictionary's role in processing compound model operations, it might be insightful to explore the implementation details of the functions associated with each operator, such as '_cdot' and '_cstack', to understand how they handle nested CompoundModels. Additionally, examining test cases that specifically target the behavior of these operators with nested models could provide further insight into the issue.",
#   "value": 75
# }

# In the Stage 4,
# Action:{"action": {"thoughts": "I need to examine the actual definitions of the operators in the '_operators' dictionary to understand how compound models are processed when computing the separability matrix.", "file_pattern": null, "code_snippet": "_operators ="}, "action_type": "FindCodeSnippet"}
# Result: {
#   "description": "The last executed action successfully identified the code snippet defining the `_operators` dictionary within the `separable.py` file. This is an essential part of the codebase as it maps modeling operators to functions that compute the relationship of axes as an array of 0-es and 1-s, which is directly related to the problem described. The output of this action is relevant and provides the necessary context to understand the behavior of the `separability_matrix` function and why it might not compute separability correctly for nested CompoundModels.",
#   "feedback": "As an alternative approach, consider analyzing the `_separable` function within the `separable.py` file. This function performs the actual computation for the separability matrix and may be where the issue with nested CompoundModels arises. Investigate how this function handles different types of CompoundModels and whether the logic needs adjustment for nested structures.",
#   "value": 90
# }

# In the Stage 5,
# Action:{"action_type": "StringReplace", "action": {"thoughts": "The issue with the separability_matrix in nested CompoundModels seems to be related to the operator definitions in the _operators dictionary. Specifically, the '&' operator is defined with _cstack function, which might not handle nested CompoundModels correctly. I'll change the definition of the '&' operator to a new function that can handle nested CompoundModels properly.", "path": "astropy/modeling/separable.py", "old_str": "_operators = {'&': _cstack, '|': _cdot, '+': _arith_oper, '-': _arith_oper,\n                '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}", "new_str": "_operators = {'&': _cstack_nested, '|': _cdot, '+': _arith_oper, '-': _arith_oper,\n                '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}"}}
# Result: {
#   "description": "The last executed action attempted to use the StringReplace function to modify the `_operators` dictionary within the separable.py file. The intention was to replace the function associated with the '&' operator to address the issue of nested CompoundModels not being handled correctly by the `separability_matrix` function. However, the action failed because the provided 'old_str' did not match the exact indentation of the original code. The agent's action is on the right track as it targets the core of the problem, but it needs to ensure the 'old_str' matches the file's exact content, including whitespace and indentation, to successfully apply the change.",
#   "feedback": "To pursue an alternative strategy, consider examining the logic within the '_cstack' function that handles the '&' operator for CompoundModels. Investigate how it processes nested models and identify any logical flaws or missing considerations that might be causing the incorrect computation of separability. This could involve debugging by adding print statements to trace how nested CompoundModels are processed or by carefully reviewing the mathematical logic that determines separability.",
#   "value": 25
# }

# In the Stage 6,
# Action:{"action": {"thoughts": "To understand how the '_operators' dictionary is used to determine the separability of compound models, I need to find the functions that handle the '&' and '|' operators. These operators are key to understanding how nested compound models are being treated in terms of separability.", "function_name": "_operators['&']", "class_name": null}, "action_type": "FindFunction"}
# Result: {
#   "description": "The action of searching for the function _operators['&'] was appropriate because it directly relates to understanding how the '&' operator affects the separability of compound models within the context of the issue. However, the search yielded no results, indicating that the function may not be explicitly named in the codebase or that the search parameters were not correctly specified. Given that the '_operators' dictionary was referenced in the code context and is central to handling compound model separability, the search was relevant but did not yield a successful outcome. The agent should consider alternative ways to explore the implementation of the '&' operator's effect on separability, potentially by searching for related functions or patterns that handle compound models.",
#   "feedback": "Since the direct search for '_operators['&']' did not yield results, a more effective approach might be to search for functions or patterns that are associated with the handling of compound models and the '&' operator. Consider using a SemanticSearch with a query related to 'handling of compound models with '&' operator' to find relevant functions or code sections that might implement the logic for handling the '&' operator within compound models. This approach can uncover the underlying functions that determine the separability behavior when '&' is used, which is crucial for diagnosing the issue.",
#   "value": 50
# }

# In the Stage 7,
# Action:{"action": {"files": [{"end_line": 309, "file_path": "astropy/modeling/separable.py", "span_ids": ["_separable"], "start_line": 306}]}, "action_type": "ViewCode", "thoughts": "To understand how the _separable function handles nested CompoundModels, I will examine the code segment specifically dealing with CompoundModels within the _separable function."}
# Result: {
#   "description": "The last executed action was to view the code for the specific section of the _separable function that deals with CompoundModels. This is the correct section to review to understand why nested CompoundModels are not being evaluated correctly for separability. Viewing this code directly is relevant and necessary for diagnosing the issue and understanding the logic that may be causing the problem with nested CompoundModels. It is also efficient as it focuses on the specific part of the function that affects CompoundModels.",
#   "feedback": "An alternative approach could be to test the _separable function with different combinations of CompoundModels to see if the issue is consistent or specific to certain types of nested models. This could provide more insight into whether the problem lies in the logic of the _separable function or in how the CompoundModels are being constructed and passed to the function. Additionally, reviewing the documentation and any existing tests for the _separable function could help identify any edge cases or special handling that may be relevant to this issue.",
#   "value": 100
# }





failed_example = '''
Repository: 
astropy\n
Issue:
Problem statement:
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
       [False,  True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True, False],
       [False, False, False,  True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True,  True, False, False],
       [ True,  True, False, False],
       [False, False,  True,  True],
       [False, False,  True,  True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?


Trajectory1:
In the Stage 1,
Action:{"action": {"thoughts": "To address the issue reported regarding the `separability_matrix` function not computing separability correctly for nested CompoundModels, I will first locate the implementation of the `separability_matrix` function. This will help me understand the logic it uses and identify any potential bugs or miscalculations with how it handles nested CompoundModels.", "query": "separability_matrix implementation", "category": "implementation"}, "action_type": "SemanticSearch"}
Result: {
  "description": "The last executed action was a SemanticSearch with the query 'separability_matrix implementation'. This search was appropriate as it directly targets the implementation of the 'separability_matrix' function, which is central to the problem described. The search is expected to yield relevant results that can help identify why the function behaves unexpectedly with nested CompoundModels. The outcome of this search provides the implementation details of the function, which is a good starting point for debugging and understanding its logic. It is on the right path to resolving the issue by allowing the agent to examine the actual code that might contain the bug.",
  "feedback": "As an alternative approach, the agent could have used a FindFunction action to directly locate the 'separability_matrix' function. This would be a more direct method if the agent already knows the exact name of the function it wants to find. However, given that the agent opted for a SemanticSearch, the next step could be to ViewCode to scrutinize the implementation details of the 'separability_matrix' function, particularly focusing on how it handles nested CompoundModels. This could provide insights into the discrepancy observed in the separability matrix for nested CompoundModels.",
  "value": 90
}

In the Stage 2,
Action:{"action_type":"ViewCode","action":{"thoughts":"To understand how the `separability_matrix` function is implemented and identify possible issues with nested CompoundModels, I'll examine the implementation in `astropy/modeling/separable.py`.","files":[{"file_path":"astropy/modeling/separable.py","start_line":null,"end_line":null,"span_ids":["separability_matrix"]}]}}
Result: {
  "description": "The last executed action was to view the implementation of the `separability_matrix` function in the file 'astropy/modeling/separable.py'. The code provided is the relevant portion of the implementation, which is directly related to the problem statement and necessary for further analysis. The function `separability_matrix` is called to compute the correlation between outputs and inputs of a model, specifically in the context of determining if transformations are separable. This is crucial for addressing the issue described in the initial problem statement, where the behavior of this function is not as expected for nested CompoundModels. Therefore, the action of viewing this code is highly relevant and necessary for diagnosing and potentially fixing the issue.",
  "feedback": "An alternative approach to diagnosing the issue could involve running specific test cases that isolate the behavior of the `separability_matrix` function with various model configurations. This could help identify edge cases or particular scenarios where the function behaves incorrectly. Additionally, a review of the CompoundModel class and how it interacts with the `separability_matrix` function could provide insights into whether the implementation of the matrix computation needs to account for nested models differently.",
  "value": 95
}

In the Stage 3,
Action:{"action": {"thoughts": "To investigate the issue with the separability_matrix not computing separability correctly for nested CompoundModels, I need to examine the implementation of the _separable function that is called within the separability_matrix function.", "file_pattern": null, "function_name": "_separable", "class_name": null}, "action_type": "FindFunction"}
Result: {
  "description": "The last executed action was to find the function `_separable`, which is a part of the implementation of the `separability_matrix`. This function is critical for determining the separability of outputs in compound models. Given the context of the problem, which involves nested CompoundModels and their separability, finding this function is a logical and appropriate step. It allows the agent to examine how separability is computed within the compound model's framework, specifically looking at how nested models affect the separability matrix.",
  "feedback": "An alternative approach could involve reviewing the compound model's composition and how it structures the left and right parts of the model. The agent could also explore how the `_operators` dictionary maps different operations between the left and right models and affects the overall separability. It would be beneficial to understand the role of each operator in the context of the separability matrix computation.",
  "value": 85
}

In the Stage 4,
Action:{
  "action": {
    "thoughts": "I need to review the _separable function implementation to understand how it calculates separability for nested CompoundModels.",
    "files": [
      {
        "file_path": "astropy/modeling/separable.py",
        "start_line": null,
        "end_line": null,
        "span_ids": ["_separable"]
      }
    ]
  },
  "action_type": "ViewCode"
}
Result: {
  "description": "The last executed action involved viewing the code for the `_separable` function within the 'astropy/modeling/separable.py' file. This function is crucial for determining the separability of outputs in compound models, which is the core issue reported by the user. The function's code was presented in its entirety, with comments for context, which is appropriate for understanding the logic behind the computation of the separability matrix.",
  "feedback": "An alternative approach would be to investigate the `_calculate_separability_matrix` method and the `_operators` dictionary within the same file. These components are likely involved in the calculation and could provide insights into why nested compound models are not computing separability correctly.",
  "value": 100
}

In the Stage 5,
Action:{"action": {"thoughts": "The issue seems to be with how the _separable function handles nested CompoundModels. Let's examine its implementation in the separable.py file to identify any potential bugs or incorrect assumptions.", "files": [{"file_path": "astropy/modeling/separable.py", "start_line": null, "end_line": null, "span_ids": ["_separable"]}]}, "action_type": "ViewCode"}
Result: {
  "description": "The last executed action was to view the code for the `_separable` function within the file `astropy/modeling/separable.py`. This function is crucial for understanding the problem at hand, as it is directly related to calculating the separability of models, which is the core issue described in the problem statement. Viewing this code is an appropriate and necessary step to diagnose the bug or misunderstanding regarding the `separability_matrix` behavior for nested CompoundModels. The code viewing provides a clear look into how the function is implemented and thus is highly relevant and necessary for resolving the issue.",
  "feedback": "As an alternative approach, consider analyzing the test cases or examples that use the `separability_matrix` function. This might help identify edge cases or specific scenarios where the function behaves unexpectedly. Additionally, reviewing the documentation and any recent changes to the `_separable` function could provide insights into whether the behavior is intended or a result of a recent change.",
  "value": 100
}

In the Stage 7,
Action:{"action_type": "StringReplace", "action": {"thoughts": "Seems like the issue is with the _operators dict in the _separable function. It's mapping the '&' operator to the wrong function which is causing the incorrect computation of the separability matrix for nested CompoundModels. I'll correct the mapping for the '&' operator to the correct function.", "path": "astropy/modeling/separable.py", "old_str": "# Maps modeling operators to a function computing and represents the\n# relationship of axes as an array of 0-es and 1-s\n_operators = {\n    '&' : _multiply\n}", "new_str": "# Maps modeling operators to a function computing and represents the\n# relationship of axes as an array of 0-es and 1-s\n_operators = {\n    '&' : _combine\n}" }}
Result: {
  "description": "The last executed action attempted to replace a part of the code in `astropy/modeling/separable.py` by identifying the exact string. However, the string provided was not found in the file, suggesting that it either changed or was not accurately matched. This action is crucial as it involves modifying the operation that affects how separability of nested compound models is computed. The failure to find the string indicates an issue with the exactness of the match or recent changes to the codebase that weren’t accounted for. This step is necessary for addressing the issue, but the execution was incorrect.",
  "feedback": "To proceed correctly, the agent should first ensure the string to be replaced is accurately identified with respect to the current codebase. This might involve re-examining the file or adjusting the string to match the latest implementation. After confirming the exact string, the agent can then attempt the replacement again. Additionally, it would be wise to consider the logic behind the replacement and whether introducing a new function like `_combine` is the best approach to handle the compound model's separability matrix correctly.",
  "value": 0
}


Trajectory2:
In the Stage 1,
Action:{
  "action": {
    "thoughts": "Given the issue that the separability_matrix is not computing separability correctly for nested CompoundModels, I need to investigate the implementation of the separability_matrix function to understand why it behaves differently when CompoundModels are nested. I will start by locating the definition of the separability_matrix function to examine its logic.",
    "function_name": "separability_matrix",
    "class_name": null
  },
  "action_type": "FindFunction"
}
Result: {
  "description": "The agent correctly executed a FindFunction action targeting the 'separability_matrix' function to diagnose the issue with the computation of separability for nested CompoundModels. This action is appropriate as it allows the agent to directly examine the implementation of the function in question, which is central to understanding why the separability matrix might not be computed correctly for nested models. The function's code was successfully found, and it includes important details about its parameters, return values, and logic, which are crucial for further analysis.",
  "feedback": "Branch out and explore alternative debugging strategies such as testing the behavior of the 'separability_matrix' function with various synthetic model combinations. This would help to isolate whether the issue is with the function's logic, its handling of nested CompoundModels, or an interaction with specific model types. Additionally, consider examining the model composition rules and interactions within the CompoundModel class to understand how they might influence the separability computation.",
  "value": 90
}

In the Stage 2,
Action:{"action_type": "ViewCode", "action": {"thoughts": "I need to examine the implementation of the separability_matrix function to understand how it computes separability and identify any potential issues with nested CompoundModels.", "files": [{"file_path": "astropy/modeling/separable.py", "start_line": 66, "end_line": 102, "span_ids": ["separability_matrix"]}]}}
Result: {
  "description": "The last executed action was to view the code of the 'separability_matrix' function from 'astropy/modeling/separable.py'. This is an appropriate action because the function is central to the issue described in the problem statement, which concerns the computation of the separability matrix for nested CompoundModels. Viewing the code allows the AI to understand how it is currently implemented, which is essential for diagnosing the issue and developing a potential fix.",
  "feedback": "An alternative approach would be to look for any open issues or discussions related to this function in the repository's issue tracker or community forums. This could provide insights into whether this is a known issue, and potentially find discussions or bug reports that have already identified the problem and possibly even suggested solutions.",
  "value": 75
}

In the Stage 3,
Action:{"action_type": "FindFunction", "action": {"thoughts": "I found the implementation of the separability_matrix function. To understand how the function processes nested CompoundModels, I need to locate the definition of its helper function _separable which is not exposed and likely a private function within the file.", "file_pattern": null, "function_name": "_separable", "class_name": null}}
Result: {
  "description": "The last executed action was to find the function '_separable', which is a part of the code under scrutiny for the issue with the 'separability_matrix'. The '_separable' function is critical for understanding how separability is calculated for compound models in Astropy. The agent correctly identified the need to dive deeper into this function, which is responsible for determining the separability of outputs for a given model. This step is appropriate for addressing the problem and is on the right path to resolving the issue.",
  "feedback": "To explore alternative paths, consider analyzing the '_operators' dictionary and its effects on the separability computation. Investigate how different operators in the '_operators' dictionary (like '& ' and '|') handle compound models and influence the final separability matrix. An alternative approach could involve testing the '_separable' function with various synthetic models to uncover any unexpected behaviors.",
  "value": 90
}

In the Stage 4,
Action:{
    "action": {
        "thoughts": "I need to investigate the _separable function to understand why the nested CompoundModels are not computing separability correctly.",
        "function_name": "_separable",
        "class_name": null
    },
    "action_type": "FindFunction"
}
Result: {
  "description": "The action to find the function _separable is appropriate as it is a crucial part of the calculation for the separability matrix. The function _separable is called within separability_matrix function to compute the separability of a given model. Reviewing this function is important for understanding why the separability matrix may not be computed correctly for nested CompoundModels, as indicated by the user's problem statement. The agent is on the right path to resolving the issue by investigating this function.",
  "feedback": "An alternative approach could involve examining the _calculate_separability_matrix method invocation within the _separable function, especially since it returns NotImplemented which might suggest that every CompoundModel doesn't properly implement this method. Investigating the CompoundModel's method for computing separability could provide insights into why nested models behave differently. Additionally, exploring the _operators dictionary to understand how different operations between models affect their separability could be another insightful investigative path.",
  "value": 75
}

In the Stage 5,
Action:{"action_type": "ViewCode", "action": {"thoughts": "To examine why the separability_matrix is not computing separability correctly for nested CompoundModels, we need to look at the implementation of the _separable function.", "files": [{"file_path": "astropy/modeling/separable.py", "start_line": 290, "end_line": 312, "span_ids": ["_separable"]}]}}
Result: {
  "description": "The 'ViewCode' action was correctly executed to retrieve the specific code segment of the '_separable' function in the 'separable.py' file. This function is crucial to the problem as it is directly involved in the computation of the separability matrix which is at the core of the issue described. The output confirms that the '_separable' function is correctly identified and its code is relevant to understanding and potentially resolving the problem with the 'separability_matrix' function's behavior concerning nested 'CompoundModels'. It is appropriate to view this code in detail to analyze the implementation and identify potential sources of the issue.",
  "feedback": "As an alternative approach, consider examining the '_operators' dictionary that maps modeling operators to functions representing the relationship of axes. This could provide insights into how different operators affect the computation of the separability matrix, especially for nested models. Understanding how these operators are applied in the '_separable' function when dealing with compound models could reveal why the matrix is not computed correctly for nested structures.",
  "value": 85
}

In the Stage 6,
Action:{"action": {"thoughts": "I need to review the implementation of the _separable function to investigate the reported issue with compound models.", "file_pattern": null, "function_name": "_separable", "class_name": null}, "action_type": "FindFunction"}
Result: {
  "description": "The last executed action was to find the function _separable within the separable.py file, which is part of the astropy.modeling package. This function is crucial for understanding and potentially fixing the issue with the separability_matrix computation for nested CompoundModels. The action is appropriate as it targets the function that computes the separability of a model, which directly relates to the reported problem.\n\nGiven the context and the fact that the function _separable is indeed part of the codebase and seems to be responsible for the logic that determines separability, the action is on the right path to resolving the issue. The code snippet provided offers insight into how separability is computed, particularly for CompoundModels, which is exactly what the problem statement is about.",
  "feedback": "As an alternative approach, consider examining the _calculate_separability_matrix method mentioned in the _separable function, which could provide more details on how the separability matrix is calculated. Additionally, looking into how _operators handle different operations could offer insights into why nested CompoundModels might not be computed correctly. Exploring the tests for the separability_matrix function might also reveal more about the expected behavior and help in identifying discrepancies.",
  "value": 80
}


Output:\n
{
  "for_all_instances": [
    "1. Relying on automated string replacement without verifying the current file content can lead to failed modifications. It is crucial to inspect the file first (e.g., using ViewCode) to confirm that the expected string pattern still exists.",
    "2. While search actions like SemanticSearch, FindFunction, and ViewCode are effective for locating and understanding relevant code sections, repeatedly using them without progressing to a verified change can result in redundant steps that do not resolve the underlying issue."
  ],
  "for_instance": [
    "1. In the astropy repository, the attempt to correct the separability_matrix issue by replacing the operator mapping in the _separable function failed because the specified string did not match the current implementation. This suggests that the expected code snippet may have changed, indicating the need for a more adaptive approach.",
    "2. The investigation into nested CompoundModels revealed that while the diagnostic steps successfully pinpointed potential areas of concern (such as the _operators dictionary), the final modification step did not apply due to inaccurate assumptions about the code structure specific to astropy."
  ]
}
'''



# 这是比较多个失败的traj的system prompt
reflection_system_prompt_1 = f'''
As a software engineering expert, you will be given multiple trajectories and an issue. Your task is to compare these trajectories, summarize experiences from the failed ones, and relate them to the given issue.

Experiences typically refer to the reasons for the failure and success of tool calls in the repository, as well as the reasons why a trajectory failed. When summarizing your experiences, carefully compare the failed trajectories with the issue. Remember that the end result of these trajectories is failure! Do not consider the final wrong result as a valid experience.

Your output should strictly follow the JSON format with the following structure:

{{
  "for_all_instances": [
    "1. ...",
    ...
  ],
  "for_instance": [
    "1. ...",
    ...
  ]
}}

If you believe an experience is relevant to all repositories and not just limited to a specific one, please write it under "for_all_repositories". If an experience is only applicable to a specific instance, write it under "for_instance". Note that the content in the two parts should NOT have any repetitions.

An example is given below:
{failed_example}
'''

user_prompt = '''
Repository: 
{repo}\n
Issue:
{issue}\n
Trajectory1:
{traj1}\n

Trajectory2:
{traj2}\n
Output:
'''