from litellm import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)


_ENTITY_LOCALIZATION_DESCRIPTION = """
Performs entity-based code localization using a three-stage pipeline:
1. Searches for relevant code snippets based on context
2. Extracts the most relevant entities from found code snippets  
3. Generates localization chains through graph traversal for each entity

This tool is designed to find the most relevant code locations for solving a specific issue by following dependency relationships in the codebase.
"""

_ENTITY_LOCALIZATION_PARAMETERS = {
    'type': 'object',
    'properties': {
        'context': {
            'description': (
                'Context string describing what to search for. This should contain keywords, '
                'functionality descriptions, or specific terms related to the issue you want to solve.'
            ),
            'type': 'string',
        },
        'max_depth': {
            'description': (
                'Maximum depth for graph traversal when generating localization chains. '
                'Higher values explore more relationships but may take longer. Default is 5.'
            ),
            'type': 'integer',
            'default': 5,
        },
        'model_name': {
            'description': (
                'LLM model to use for entity extraction. Default is "azure/gpt-4o".'
            ),
            'type': 'string',
            'default': 'azure/gpt-4o',
        },
    },
    'required': ['context'],
}

EntityLocalizationTool = ChatCompletionToolParam(
    type='function',
    function=ChatCompletionToolParamFunctionChunk(
        name='run_entity_localization_pipeline',
        description=_ENTITY_LOCALIZATION_DESCRIPTION,
        parameters=_ENTITY_LOCALIZATION_PARAMETERS,
    ),
)


def format_localization_results(results: dict) -> str:
    """
    Format the results of entity localization pipeline for display.
    
    Args:
        results: Results dictionary from the pipeline
        
    Returns:
        Formatted string representation of the results
    """
    output = []
    
    output.append("=== Entity Localization Results ===")
    output.append(f"Instance: {results['instance_id']}")
    output.append(f"Context: {results['context']}")
    output.append("")
    
    # Show extracted entities
    entities = results.get('extracted_entities', [])
    output.append(f"Found {len(entities)} relevant entities:")
    for i, entity in enumerate(entities):
        output.append(f"  {i+1}. {entity['entity_id']} ({entity['entity_type']})")
        output.append(f"     Relevance: {entity['relevance_reason']}")
    output.append("")
    
    # Show localization chains
    chains = results.get('localization_chains', [])
    output.append(f"Generated {len(chains)} localization chains:")
    
    for i, chain_data in enumerate(chains):
        start_entity = chain_data['start_entity']
        chain = chain_data['chain']
        
        output.append(f"\n  Chain {i+1}: Starting from {start_entity['entity_id']}")
        output.append(f"  Chain length: {len(chain)} steps")
        
        if chain:
            output.append("  Localization path:")
            current_entity = None
            
            for step in chain:
                if 'entity_id' in step:
                    # This is an entity step
                    current_entity = step['entity_id']
                    output.append(f"    [{step['depth']}] {step['entity_id']} ({step['entity_type']})")
                    
                elif 'from_entity' in step and 'to_entity' in step:
                    # This is an edge step
                    direction_symbol = "->" if step['direction'] == 'downstream' else "<-"
                    output.append(f"    {direction_symbol} {step['edge_type']} {direction_symbol}")
    
    return "\n".join(output)
