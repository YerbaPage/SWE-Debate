import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
from typing import Dict, List, Tuple, Any
import logging

class LocalizationChainEmbedding:
    """定位链嵌入计算器"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", 
                 cache_dir: str = '/data/swebench/workspace_agentless/Agentless/models'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """平均池化"""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """构建详细指令"""
        return f'Instruct: {task_description}\nQuery: {query}'

    def chain_to_text(self, chain: List[str]) -> str:
        """将定位链转换为文本表示"""
        if not chain:
            return "empty_chain"
        
        # 将entity ID列表转换为可读的文本描述
        chain_parts = []
        for entity_id in chain:
            # 提取文件名和实体名
            if ':' in entity_id:
                file_part, entity_part = entity_id.split(':', 1)
                chain_parts.append(f"file:{file_part} entity:{entity_part}")
            else:
                chain_parts.append(f"file:{entity_id}")
        
        return " -> ".join(chain_parts)

    def compute_chain_embeddings(self, chains: List[List[str]], 
                                task_description: str = None) -> Tensor:
        """
        计算定位链的嵌入向量
        
        Args:
            chains: 定位链列表，每个定位链是entity ID的列表
            task_description: 任务描述
            
        Returns:
            嵌入向量张量
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
        
        # 将定位链转换为文本
        chain_texts = []
        for i, chain in enumerate(chains):
            chain_text = self.chain_to_text(chain)
            if i == 0:  # 第一个作为查询
                chain_texts.append(self.get_detailed_instruct(task_description, chain_text))
            else:  # 其他作为文档
                chain_texts.append(chain_text)
        
        # 令牌化
        batch_dict = self.tokenizer(chain_texts, max_length=1024, padding=True, 
                                   truncation=True, return_tensors='pt')
        
        # 计算嵌入
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        # 标准化嵌入
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def select_diverse_chains(self, chains: List[List[str]], k: int = 4) -> Tuple[List[int], List[float]]:
        """
        选择多样化的定位链（基于最长链）
        
        Args:
            chains: 定位链列表
            k: 选择的链数量（不包括最长链）
            
        Returns:
            选中的链索引和相似度分数
        """
        if not chains:
            return [], []
        
        # 1. 过滤空链并记录索引映射
        valid_chains = []
        valid_indices = []
        for i, chain in enumerate(chains):
            if chain and len(chain) > 0:  # 过滤空链
                valid_chains.append(chain)
                valid_indices.append(i)
        
        if not valid_chains:
            logging.warning("所有定位链都为空")
            return [], []
        
        logging.info(f"过滤空链后，有效定位链数量: {len(valid_chains)} (原始: {len(chains)})")
        
        # 2. 去重：使用字符串表示作为去重键
        unique_chains = []
        unique_indices = []
        seen_chains = set()
        
        for i, chain in enumerate(valid_chains):
            # 将链转换为字符串作为去重键
            chain_key = str(sorted(chain)) if isinstance(chain, list) else str(chain)
            if chain_key not in seen_chains:
                seen_chains.add(chain_key)
                unique_chains.append(chain)
                unique_indices.append(valid_indices[i])  # 映射回原始索引
        
        logging.info(f"去重后，唯一定位链数量: {len(unique_chains)} (过滤前: {len(valid_chains)})")
        
        if not unique_chains:
            logging.warning("去重后没有剩余的定位链")
            return [], []
            
        # 3. 找到最长的链
        chain_lengths = [len(chain) for chain in unique_chains]
        longest_idx_in_unique = chain_lengths.index(max(chain_lengths))
        longest_original_idx = unique_indices[longest_idx_in_unique]
        longest_chain = unique_chains[longest_idx_in_unique]
        
        logging.info(f"最长定位链索引: {longest_original_idx} (原始), 长度: {len(longest_chain)}")
        logging.info(f"最长定位链: {longest_chain}")
        
        if len(unique_chains) <= 1:
            return [longest_original_idx], [1.0]
        
        # 4. 计算嵌入（最长链作为查询，其他作为候选）
        other_chains = [unique_chains[i] for i in range(len(unique_chains)) if i != longest_idx_in_unique]
        all_chains_for_embedding = [longest_chain] + other_chains
        embeddings = self.compute_chain_embeddings(all_chains_for_embedding)
        
        # 5. 计算相似度分数
        query_embedding = embeddings[0:1]  # 最长链的嵌入
        doc_embeddings = embeddings[1:]    # 其他链的嵌入
        
        similarities = (query_embedding @ doc_embeddings.T).squeeze().tolist()
        if isinstance(similarities, float):  # 只有一个其他链的情况
            similarities = [similarities]
        
        # 6. 创建索引映射（排除最长链的索引）
        other_original_indices = [unique_indices[i] for i in range(len(unique_chains)) if i != longest_idx_in_unique]
        
        # 7. 按相似度排序（最不相似的在前）
        indexed_similarities = list(zip(other_original_indices, similarities))
        indexed_similarities.sort(key=lambda x: x[1])  # 升序排列，最不相似的在前
        
        # 8. 选择最不相似的k条链
        selected_indices = [longest_original_idx]  # 总是包含最长链
        selected_scores = [1.0]  # 最长链的"相似度"设为1.0
        
        for i, (chain_idx, similarity) in enumerate(indexed_similarities):
            if len(selected_indices) >= k + 1:  # k条不相似的链 + 1条最长链
                break
            selected_indices.append(chain_idx)
            selected_scores.append(similarity)
        
        logging.info(f"选择的定位链索引: {selected_indices}")
        logging.info(f"对应的相似度分数: {selected_scores}")
        
        return selected_indices, selected_scores

# 保持原有的screening函数以兼容性
def screening(pre_issues: Dict, cur_issue: Dict,
              tokenizer=None, model=None, k=10):
    """原有的issue筛选函数，保持兼容性"""
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