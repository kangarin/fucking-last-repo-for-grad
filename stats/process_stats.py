import pandas as pd
import numpy as np

def calculate_metrics(csv_file_path, queue_threshold, max_queue_length):
    """
    计算监控指标，包括综合准确度、处理时延和奖励函数
    
    参数:
        csv_file_path: CSV文件路径
        queue_threshold: 队列阈值，默认为10
        max_queue_length: 最大队列长度，默认为50
    
    返回:
        包含计算指标的字典
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
    
    # 计算平均指标
    metrics = {
        'avg_combined_accuracy': df['combined_accuracy'].mean(),
        'avg_processing_latency': df['processing_latency'].mean(),
        'avg_total_latency': df['total_latency'].mean(),
        'avg_reward': df['reward'].mean(),
        'total_records': len(df)
    }
    
    return metrics, df

def print_metrics_report(metrics):
    """打印指标报告"""
    print("\n==== 性能指标报告 ====")
    print(f"综合准确度: {metrics['avg_combined_accuracy']:.2f}")
    print(f"平均处理时延: {metrics['avg_processing_latency']:.2f} s")
    print(f"平均总时延: {metrics['avg_total_latency']:.2f} s")
    print(f"平均奖励值: {metrics['avg_reward']:.2f}")
    print(f"总记录数: {metrics['total_records']}")
    print("=====================")

def main():
    # 配置参数
    csv_file_path = 'stats/stats.csv'  # 请修改为实际的CSV文件路径
    queue_threshold = 10
    max_queue_length = 50
    
    try:
        # 计算指标
        metrics, processed_df = calculate_metrics(csv_file_path, queue_threshold, max_queue_length)
        
        # 打印报告
        print_metrics_report(metrics)
        
        # # 可选：保存处理后的数据到新CSV
        # processed_df.to_csv('processed_metrics.csv', index=False)
        # print(f"详细数据已保存至 'processed_metrics.csv'")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    main()