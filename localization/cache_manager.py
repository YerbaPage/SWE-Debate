#!/usr/bin/env python3
"""
Entity Localization Pipeline 缓存管理工具

这个工具提供了管理和分析pipeline缓存的功能，包括：
1. 查看所有缓存实例
2. 分析缓存文件内容
3. 比较不同运行的结果
4. 清理和导出缓存

使用方法:
python cache_manager.py [command] [options]

作者：基于entity_localization_pipeline_original_huawei.py
"""

import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
from entity_localization_pipeline_stage7_runner import Stage7Runner


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self.runner = Stage7Runner()
        self.cache_dir = self.runner.cache_dir
        
    def list_all_instances(self) -> Dict[str, Any]:
        """列出所有缓存实例的详细信息"""
        instances = self.runner.list_cached_instances()
        
        result = {
            'total_instances': len(instances),
            'instances': {}
        }
        
        for instance_id in instances:
            cache_files = self.runner.list_cached_files_for_instance(instance_id)
            result['instances'][instance_id] = {
                'cache_files_count': len(cache_files),
                'cache_files': cache_files,
                'latest_cache': cache_files[0] if cache_files else None,
                'summary': self.runner.get_cache_summary(instance_id)
            }
        
        return result
    
    def analyze_instance(self, instance_id: str, cache_file: str = None) -> Dict[str, Any]:
        """分析单个实例的详细信息"""
        if cache_file is None:
            cache_files = self.runner.list_cached_files_for_instance(instance_id)
            if not cache_files:
                return {'error': '未找到缓存文件'}
            cache_files.sort(reverse=True)
            cache_file = cache_files[0]
        
        cache_file_path = os.path.join(self.cache_dir, instance_id, cache_file)
        
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            analysis = {
                'instance_id': instance_id,
                'cache_file': cache_file,
                'file_path': cache_file_path,
                'file_size_mb': os.path.getsize(cache_file_path) / (1024 * 1024),
                'stages': {}
            }
            
            # 分析每个stage
            for stage_name, stage_info in cache_data.items():
                stage_analysis = {
                    'timestamp': stage_info.get('timestamp'),
                    'data_keys': list(stage_info.get('data', {}).keys()),
                    'data_size': len(str(stage_info.get('data', {})))
                }
                
                # 提取特定stage的关键信息
                data = stage_info.get('data', {})
                
                if 'initial_entities' in data:
                    stage_analysis['initial_entities_count'] = len(data['initial_entities'])
                    stage_analysis['initial_entities'] = data['initial_entities']
                
                if 'entity_groups' in data:
                    stage_analysis['entity_groups_count'] = len(data['entity_groups'])
                    stage_analysis['total_related_entities'] = data.get('total_related_entities', 0)
                
                if 'all_chains' in data:
                    stage_analysis['total_chains'] = len(data['all_chains'])
                
                if 'selected_chains' in data:
                    stage_analysis['selected_chains_count'] = len(data['selected_chains'])
                
                if 'voting_result' in data:
                    voting_result = data['voting_result']
                    stage_analysis['winning_chain_id'] = voting_result.get('winning_chain_id')
                    stage_analysis['voting_success'] = voting_result.get('success')
                
                if 'first_round_analyses' in data:
                    stage_analysis['agents_count'] = len(data['first_round_analyses'])
                
                if 'final_plan' in data:
                    final_plan = data['final_plan']
                    if 'final_plan' in final_plan:
                        modifications = final_plan['final_plan'].get('modifications', [])
                        stage_analysis['modification_steps'] = len(modifications)
                
                analysis['stages'][stage_name] = stage_analysis
            
            return analysis
            
        except Exception as e:
            return {'error': f'分析实例失败: {e}'}
    
    def compare_runs(self, instance_id: str, cache_files: List[str] = None) -> Dict[str, Any]:
        """比较同一实例的多次运行结果"""
        if cache_files is None:
            cache_files = self.runner.list_cached_files_for_instance(instance_id)
        
        if len(cache_files) < 2:
            return {'error': '需要至少2个缓存文件进行比较'}
        
        comparison = {
            'instance_id': instance_id,
            'compared_files': cache_files,
            'differences': {},
            'summary': {}
        }
        
        analyses = {}
        for cache_file in cache_files:
            analyses[cache_file] = self.analyze_instance(instance_id, cache_file)
        
        # 比较关键指标
        metrics = [
            'initial_entities_count',
            'total_related_entities', 
            'total_chains',
            'selected_chains_count',
            'winning_chain_id',
            'modification_steps'
        ]
        
        for metric in metrics:
            values = {}
            for cache_file, analysis in analyses.items():
                value = None
                for stage_name, stage_data in analysis.get('stages', {}).items():
                    if metric in stage_data:
                        value = stage_data[metric]
                        break
                values[cache_file] = value
            
            # 检查是否有差异
            unique_values = set(v for v in values.values() if v is not None)
            if len(unique_values) > 1:
                comparison['differences'][metric] = values
        
        comparison['summary'] = {
            'total_differences': len(comparison['differences']),
            'files_compared': len(cache_files)
        }
        
        return comparison
    
    def export_cache(self, instance_id: str, cache_file: str = None, 
                    output_file: str = None) -> str:
        """导出缓存到文件"""
        if cache_file is None:
            cache_files = self.runner.list_cached_files_for_instance(instance_id)
            if not cache_files:
                raise ValueError('未找到缓存文件')
            cache_files.sort(reverse=True)
            cache_file = cache_files[0]
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{instance_id}_{timestamp}_export.json"
        
        cache_file_path = os.path.join(self.cache_dir, instance_id, cache_file)
        
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # 添加导出元数据
        export_data = {
            'export_metadata': {
                'instance_id': instance_id,
                'source_cache_file': cache_file,
                'export_timestamp': datetime.now().isoformat(),
                'export_tool': 'cache_manager.py'
            },
            'cache_data': cache_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def clean_old_caches(self, instance_id: str = None, keep_latest: int = 3) -> Dict[str, Any]:
        """清理旧的缓存文件，保留最新的几个"""
        if instance_id:
            instances = [instance_id]
        else:
            instances = self.runner.list_cached_instances()
        
        result = {
            'cleaned_files': 0,
            'kept_files': 0,
            'details': {}
        }
        
        for inst_id in instances:
            cache_files = self.runner.list_cached_files_for_instance(inst_id)
            cache_files.sort(reverse=True)  # 最新的在前
            
            files_to_keep = cache_files[:keep_latest]
            files_to_delete = cache_files[keep_latest:]
            
            deleted_count = 0
            for file_to_delete in files_to_delete:
                file_path = os.path.join(self.cache_dir, inst_id, file_to_delete)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    logging.error(f"删除文件失败 {file_path}: {e}")
            
            result['details'][inst_id] = {
                'kept': len(files_to_keep),
                'deleted': deleted_count,
                'kept_files': files_to_keep
            }
            
            result['cleaned_files'] += deleted_count
            result['kept_files'] += len(files_to_keep)
        
        return result


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='Entity Localization Pipeline 缓存管理工具')
    parser.add_argument('command', choices=['list', 'analyze', 'compare', 'export', 'clean'],
                       help='要执行的命令')
    parser.add_argument('--instance', '-i', type=str, help='实例ID')
    parser.add_argument('--cache-file', '-c', type=str, help='缓存文件名')
    parser.add_argument('--output', '-o', type=str, help='输出文件名')
    parser.add_argument('--keep', type=int, default=3, help='清理时保留的文件数量')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = CacheManager()
    
    try:
        if args.command == 'list':
            result = manager.list_all_instances()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.command == 'analyze':
            if not args.instance:
                print("错误: analyze命令需要指定--instance参数")
                return
            
            result = manager.analyze_instance(args.instance, args.cache_file)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.command == 'compare':
            if not args.instance:
                print("错误: compare命令需要指定--instance参数")
                return
            
            result = manager.compare_runs(args.instance)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.command == 'export':
            if not args.instance:
                print("错误: export命令需要指定--instance参数")
                return
            
            output_file = manager.export_cache(args.instance, args.cache_file, args.output)
            print(f"缓存已导出到: {output_file}")
        
        elif args.command == 'clean':
            result = manager.clean_old_caches(args.instance, args.keep)
            print(json.dumps(result, indent=2, ensure_ascii=False))
    
    except Exception as e:
        print(f"错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
