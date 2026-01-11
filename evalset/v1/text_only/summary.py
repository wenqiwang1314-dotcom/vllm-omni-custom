import re
import ast
import pandas as pd

def extract_inference_times(log_file_path):
    """
    从 vLLM Omni 日志中提取分阶段的推理耗时和Token统计。
    """
    data_records = []
    
    # 用于捕获 [Summary] 字典内容的正则表达式
    # 匹配日志前缀，并提取主要的消息部分
    log_prefix_pattern = re.compile(r"\[.*?\]\s+.*?INFO.*?\[.*?:.*?\]\s+(.*)")
    
    current_dict_str = ""
    is_capturing = False

    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 识别 Summary 块的开始
            if "[Summary] {" in line:
                is_capturing = True
                current_dict_str = ""
            
            if is_capturing:
                # 提取去除日志前缀后的内容
                match = log_prefix_pattern.search(line)
                if match:
                    content = match.group(1)
                    # 如果是第一行，去掉 "[Summary] " 前缀
                    if content.startswith("[Summary] "):
                        content = content[len("[Summary] "):]
                    
                    current_dict_str += content
                    
                    # 检查字典是否结束（简单的启发式检查：以 '}' 结尾）
                    if content.strip().endswith("}"):
                        try:
                            # 将字符串解析为 Python 字典
                            log_dict = ast.literal_eval(current_dict_str)
                            
                            # 提取关键信息
                            stages = log_dict.get('stages', [])
                            record = {
                                'timestamp': line[1:20], # 从日志行首提取时间戳
                                'e2e_latency_ms': log_dict.get('e2e_total_time_ms'),
                            }
                            
                            # 遍历各阶段提取数据
                            for stage in stages:
                                s_id = stage.get('stage_id')
                                # 根据 Stage ID 命名列
                                if s_id == 0:
                                    stage_name = "Thinker"
                                elif s_id == 1:
                                    stage_name = "Talker"
                                elif s_id == 2:
                                    stage_name = "Audio"
                                else:
                                    stage_name = f"Stage{s_id}"
                                
                                record[f'{stage_name}_Latency_ms'] = stage.get('total_time_ms')
                                record[f'{stage_name}_Tokens'] = stage.get('tokens')
                                record[f'{stage_name}_TPS'] = stage.get('avg_tokens_per_s')

                            data_records.append(record)
                            
                            # 重置状态
                            is_capturing = False
                            current_dict_str = ""
                        except (ValueError, SyntaxError):
                            # 解析失败可能是因为跨行字典还没结束，继续读取下一行
                            pass
                else:
                    # 如果无法匹配前缀，可能需要停止捕获或做其他处理
                    pass

    # 转换为 DataFrame 并整理格式
    df = pd.DataFrame(data_records)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # 调整列顺序
        cols = ['timestamp', 'e2e_latency_ms'] + [c for c in df.columns if c not in ['timestamp', 'e2e_latency_ms']]
        df = df[cols]
        
    return df

# 使用示例
if __name__ == "__main__":
    file_path = '/home/konnext/Lucas/vllm-omni/evalset/v1/logs/vllm_omni_2026-01-10_113221.log'
    df_result = extract_inference_times(file_path)
    
    # 打印前5行数据
    print(df_result.head().to_markdown(index=False))
    
    # 保存结果到 CSV
    df_result.to_csv('omni_inference_metrics.csv', index=False)
    print("\n结果已保存至 omni_inference_metrics.csv")
