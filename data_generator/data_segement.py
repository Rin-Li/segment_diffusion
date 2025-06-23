import torch
import numpy as np


class PlaneDataSegment:
    def __init__(self, number_map, relate_length, each_path_length, each_map_length, data_path):
        self.number_map = number_map
        self.relate_length = relate_length
        self.data_path = data_path
        self.each_map_length = each_map_length
        self.each_path_length = each_path_length
        
        self.data_set = np.load(data_path, allow_pickle=True)
        self.data_set = self.data_set.item()
        
        self.start = self.data_set['start']
        self.goal = self.data_set['goal']
        self.paths = self.data_set['paths']
        self.map = self.data_set['map']
        
        # 存储分割后的数据
        self.segmented_data = {
            'start_segments': [],
            'middle_segments': [],
            'end_segments': []
        }
    
    def handle_start_map(self, map_segment, start, path_segment):
        """处理起始地图段"""
        self.segmented_data['start_segments'].append({
            'map': map_segment,
            'start': start,
            'path': path_segment
        })
    
    def handle_middle_map(self, map_segment, path_segment):
        """处理中间地图段"""
        self.segmented_data['middle_segments'].append({
            'map': map_segment,
            'path': path_segment
        })
    
    def handle_end_map(self, map_segment, goal, path_segment):
        """处理结束地图段"""
        self.segmented_data['end_segments'].append({
            'map': map_segment,
            'goal': goal,
            'path': path_segment
        })
    
    def segment_path(self, path, num_segments=3):
        """将路径分割为指定数量的段"""
        if len(path) < num_segments:
            # 如果路径点数少于段数，重复最后一点
            path = np.vstack([path, np.tile(path[-1], (num_segments - len(path), 1))])
        
        segment_length = len(path) // num_segments
        segments = []
        
        for i in range(num_segments):
            start_idx = i * segment_length
            if i == num_segments - 1:  # 最后一段包含剩余所有点
                end_idx = len(path)
            else:
                end_idx = (i + 1) * segment_length + 1  # 重叠一个点确保连续性
            
            segments.append(path[start_idx:end_idx])
        
        return segments
    
    def segment_data(self):
        for idx in range(len(self.start)):
            start = self.start[idx]
            goal = self.goal[idx]
            path = self.paths[idx]
            map_data = self.map[idx]
            
            # 确保路径长度符合要求
            if len(path) != self.each_path_length:
                # 这里可以调用RRT*的重采样方法，或者简单的线性插值
                path = self._resample_path(path, self.each_path_length)
            
            # 分割路径
            path_segments = self.segment_path(path, 3)
            
            # 处理各个段
            self.handle_start_map(
                map_data[:, :self.each_map_length], 
                start, 
                path_segments[0]
            )
            self.handle_middle_map(
                map_data[:, self.each_map_length:-self.each_map_length], 
                path_segments[1]
            )
            self.handle_end_map(
                map_data[:, -self.each_map_length:], 
                goal, 
                path_segments[2]
            )
    
    def _resample_path(self, path, target_length):
        """重采样路径到目标长度"""
        if len(path) == target_length:
            return path
        
        # 计算累积距离
        distances = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_distance = cumulative_distances[-1]
        
        # 创建等间距的采样点
        target_distances = np.linspace(0, total_distance, target_length)
        
        # 插值生成新轨迹点
        resampled_path = []
        for target_dist in target_distances:
            idx = np.searchsorted(cumulative_distances, target_dist)
            
            if idx == 0:
                resampled_path.append(path[0])
            elif idx >= len(path):
                resampled_path.append(path[-1])
            else:
                t = (target_dist - cumulative_distances[idx-1]) / (cumulative_distances[idx] - cumulative_distances[idx-1])
                point = path[idx-1] + t * (path[idx] - path[idx-1])
                resampled_path.append(point)
        
        return np.array(resampled_path)
    
    def save_segmented_data(self, output_path):
        """保存分割后的数据"""
        np.save(output_path, self.segmented_data)