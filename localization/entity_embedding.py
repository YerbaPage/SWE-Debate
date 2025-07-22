import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from typing import Dict, List, Tuple, Any
import logging

class LocalizationChainEmbedding:
    """Localization chain embedding calculator"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", 
                 cache_dir: str = '/data/swebench/workspace_agentless/Agentless/models'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Average pooling"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Build detailed instruction"""
        return f'Instruct: {task_description}\nQuery: {query}'

    def chain_to_text(self, chain: List[str]) -> str:
        """Convert localization chain to text representation"""
        if not chain:
            return "empty_chain"
        
        # Convert entity ID list to readable text description
        chain_parts = []
        for entity_id in chain:
            # Extract file name and entity name
            if ':' in entity_id:
                file_part, entity_part = entity_id.split(':', 1)
                chain_parts.append(f"file:{file_part} entity:{entity_part}")
            else:
                chain_parts.append(f"file:{entity_id}")
        
        return " -> ".join(chain_parts)

    def compute_chain_embeddings(self, chains: List[List[str]], 
                                task_description: str = None) -> Tensor:
        """
        Compute embeddings for localization chains
        
        Args:
            chains: List of localization chains, each chain is a list of entity IDs
            task_description: Task description
            
        Returns:
            Embedding tensor
        """
        if task_description is None:
            task_description = (
                "Given a code localization chain that represents a path through code entities "
                "(files, classes, functions) connected by dependencies, identify other chains "
                "that follow similar patterns, architectural flows, or functional sequences. "
                "Analyze the semantic relationships, structural patterns, and logical progression "
                "to find chains that share comparable entity types, naming conventions, "
                "dependency relationships, or problem-solving approaches."
            )
        
        # Convert localization chains to text
        chain_texts = []
        for i, chain in enumerate(chains):
            chain_text = self.chain_to_text(chain)
            if i == 0:  # First one as query
                chain_texts.append(self.get_detailed_instruct(task_description, chain_text))
            else:  # Others as documents
                chain_texts.append(chain_text)
        
        # Tokenization
        batch_dict = self.tokenizer(chain_texts, max_length=1024, padding=True, 
                                   truncation=True, return_tensors='pt')
        
        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def select_diverse_chains(self, chains: List[List[str]], k: int = 4) -> Tuple[List[int], List[float]]:
        """
        Select diverse localization chains (based on longest chain)
        
        Args:
            chains: List of localization chains
            k: Number of chains to select (excluding the longest chain)
            
        Returns:
            Selected chain indices and similarity scores
        """
        if not chains:
            return [], []
        
        valid_chains = []
        valid_indices = []
        for i, chain in enumerate(chains):
            if chain and len(chain) > 0:  
                valid_chains.append(chain)
                valid_indices.append(i)
        
        if not valid_chains:
            logging.warning("All localization chains are empty")
            return [], []
        
        logging.info(f"After filtering empty chains, valid chain count: {len(valid_chains)} (original: {len(chains)})")

        unique_chains = []
        unique_indices = []
        seen_chains = set()
        
        for i, chain in enumerate(valid_chains):
            chain_key = str(sorted(chain)) if isinstance(chain, list) else str(chain)
            if chain_key not in seen_chains:
                seen_chains.add(chain_key)
                unique_chains.append(chain)
                unique_indices.append(valid_indices[i])  # Map back to original index
        
        logging.info(f"After deduplication, unique chain count: {len(unique_chains)} (before filtering: {len(valid_chains)})")
        
        if not unique_chains:
            logging.warning("No remaining localization chains after deduplication")
            return [], []
            
        chain_lengths = [len(chain) for chain in unique_chains]
        longest_idx_in_unique = chain_lengths.index(max(chain_lengths))
        longest_original_idx = unique_indices[longest_idx_in_unique]
        longest_chain = unique_chains[longest_idx_in_unique]
        
        logging.info(f"Longest localization chain index: {longest_original_idx} (original), length: {len(longest_chain)}")
        logging.info(f"Longest localization chain: {longest_chain}")
        
        if len(unique_chains) <= 1:
            return [longest_original_idx], [1.0]
        
        other_chains = [unique_chains[i] for i in range(len(unique_chains)) if i != longest_idx_in_unique]
        all_chains_for_embedding = [longest_chain] + other_chains
        embeddings = self.compute_chain_embeddings(all_chains_for_embedding)
        
        query_embedding = embeddings[0:1]  
        doc_embeddings = embeddings[1:]  
        
        similarities = (query_embedding @ doc_embeddings.T).squeeze().tolist()
        if isinstance(similarities, float): 
            similarities = [similarities]
        
        other_original_indices = [unique_indices[i] for i in range(len(unique_chains)) if i != longest_idx_in_unique]

        indexed_similarities = list(zip(other_original_indices, similarities))
        indexed_similarities.sort(key=lambda x: x[1])  

        selected_indices = [longest_original_idx]  
        selected_scores = [1.0]  
        
        for i, (chain_idx, similarity) in enumerate(indexed_similarities):
            if len(selected_indices) >= k + 1:  
                break
            selected_indices.append(chain_idx)
            selected_scores.append(similarity)
        
        return selected_indices, selected_scores

def screening(pre_issues: Dict, cur_issue: Dict,
              tokenizer=None, model=None, k=10):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct",
                                                 cache_dir='/home/workspace/models')
    if model is None:
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct",
                                         cache_dir='/home/workspace/models')
    
    def average_pool(last_hidden_states: Tensor,
                     attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def formalize(issues):
        tmp = []
        ids = []
        for id, data in issues.items():
            ids.append(id)
            tmp.append(f"issue_type: {data['issue_type']} \ndescription: {data['description']}")
        return tmp, ids

    # Each query must come with a one-sentence instruction that describes the task
    task = "Given the prior issues, your task is to analyze a current issue's problem statement and select the most relevant prior issue that could help resolve it."
    cur_query, _ = formalize(cur_issue)
    queries = [
        get_detailed_instruct(task, cur_query[0])
    ]
    # No need to add instruction for retrieval documents
    documents, ids = formalize(issues=pre_issues)
    input_texts = queries + documents

    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=1024, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    id2score = {}
    for i in range(len(ids)):
        id2score[ids[i]] = scores.tolist()[0][i]
    id2score = sorted(id2score.items(), key=lambda x: x[1], reverse=True)
    topkids = [k for k, v in id2score[:k]]
    return id2score, topkids