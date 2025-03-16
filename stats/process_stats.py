import pandas as pd
import numpy as np

def calculate_metrics(csv_file_path, queue_threshold):
    """
    计算监控指标，包括综合准确度、处理时延和奖励函数，同时计算方差
    
    参数:
        csv_file_path: CSV文件路径
        queue_threshold: 队列阈值，默认为10
    
    返回:
        包含计算指标及其方差的字典
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 计算综合准确度 (模型精度和置信度的均值)
    df['combined_accuracy'] = (df['cur_model_accuracy'] / 100 + df['avg_confidence'])
    
    # 计算队列比率
    df['queue_ratio'] = df['queue_length'] / queue_threshold
    
    # 计算权重
    df['w_accuracy'] = np.maximum(1 - df['queue_ratio'], 0)
    df['w_latency'] = np.minimum(df['queue_ratio'], 5)
    
    # 计算奖励函数
    df['reward'] = (df['w_accuracy'] * df['combined_accuracy'] - 
                   df['w_latency'] * df['processing_latency'])
    
    # 计算平均指标和方差
    metrics = {
        'avg_combined_accuracy': df['combined_accuracy'].mean(),
        'var_combined_accuracy': df['combined_accuracy'].var(),
        'std_combined_accuracy': df['combined_accuracy'].std(),
        
        'avg_processing_latency': df['processing_latency'].mean(),
        'var_processing_latency': df['processing_latency'].var(),
        'std_processing_latency': df['processing_latency'].std(),
        
        'avg_total_latency': df['total_latency'].mean(),
        'var_total_latency': df['total_latency'].var(),
        'std_total_latency': df['total_latency'].std(),
        
        'avg_queue_length': df['queue_length'].mean(),
        'var_queue_length': df['queue_length'].var(),
        'std_queue_length': df['queue_length'].std(),
        
        'avg_reward': df['reward'].mean(),
        'var_reward': df['reward'].var(),
        'std_reward': df['reward'].std(),
        
        'total_records': len(df)
    }
    
    return metrics, df

def print_metrics_report(metrics):
    """打印指标报告，包括方差和标准差"""
    print("\n==== 性能指标报告 ====")
    print(f"综合准确度: {metrics['avg_combined_accuracy']:.4f} ± {metrics['std_combined_accuracy']:.4f}")
    print(f"平均处理时延: {metrics['avg_processing_latency']:.4f} ± {metrics['std_processing_latency']:.4f} s")
    print(f"平均总时延: {metrics['avg_total_latency']:.4f} ± {metrics['std_total_latency']:.4f} s")
    print(f"平均队列长度: {metrics['avg_queue_length']:.4f} ± {metrics['std_queue_length']:.4f}")
    print(f"平均奖励值: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}")
    print(f"总记录数: {metrics['total_records']}")
    print("\n方差:")
    print(f"综合准确度方差: {metrics['var_combined_accuracy']:.6f}")
    print(f"处理时延方差: {metrics['var_processing_latency']:.6f}")
    print(f"总时延方差: {metrics['var_total_latency']:.6f}")
    print(f"队列长度方差: {metrics['var_queue_length']:.6f}")
    print(f"奖励值方差: {metrics['var_reward']:.6f}")
    print("=====================")

def export_metrics_to_csv(metrics, output_path='metrics_summary.csv'):
    """将计算的指标导出到CSV文件"""
    # 创建单行DataFrame
    metrics_df = pd.DataFrame([metrics])
    # 保存到CSV
    metrics_df.to_csv(output_path, index=False)
    print(f"指标摘要已保存至 '{output_path}'")

def main():
    # 配置参数
    csv_file_path = 'stats/stats.csv'  # 请修改为实际的CSV文件路径
    queue_threshold = 10
    
    try:
        # 计算指标
        metrics, processed_df = calculate_metrics(csv_file_path, queue_threshold)
        
        # 打印报告
        print_metrics_report(metrics)
        
        # 保存指标摘要
        # export_metrics_to_csv(metrics)
        
        # # 可选：保存处理后的数据到新CSV
        # processed_df.to_csv('processed_metrics.csv', index=False)
        # print(f"详细数据已保存至 'processed_metrics.csv'")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()