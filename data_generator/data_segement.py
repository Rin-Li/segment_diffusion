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
    
    def handle_middle_map(self):
        pass
    
    def handle_start_map(self):
        pass
    
    def handle_end_map(self):
        pass
    
    def segment_data(self):
        for idx in range(len(self.start)):
            start = self.start[idx]
            goal = self.goal[idx]
            path = self.paths[idx]
            map_data = self.map[idx]
            self.handle_start_map(map[idx][:, :self.each_map_length], start, path)
            self.handle_middle_map(map_data[:, self.each_map_length:-self.each_map_length], path)
            self.handle_end_map(map_data[:, -self.each_map_length:], goal, path)
            
            
            
    
    
        