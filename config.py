import yaml
from pathlib import Path

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from yaml file"""
        project_root = Path(__file__).parent
        config_path = project_root / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
            
        # Convert relative paths to absolute using project root
        # Video source paths for both stream types
        for key in ['stream', 'dynamic_stream']:
            if key in self._config:
                video_path = self._config[key]['source_path']
                if not Path(video_path).is_absolute():
                    self._config[key]['source_path'] = str(project_root / video_path)
            
        # Model weights directory
        weights_dir = self._config['edge']['models']['weights_dir']
        if not Path(weights_dir).is_absolute():
            self._config['edge']['models']['weights_dir'] = str(project_root / weights_dir)
    
    def get_cloud_display_config(self):
        """Get cloud display server configuration"""
        return self._config['cloud']['display_server']
    
    def get_edge_processing_config(self):
        """Get edge processing server configuration"""
        return self._config['edge']['processing_server']
    
    def get_stream_config(self):
        """Get static stream configuration"""
        return self._config['stream']
    
    def get_dynamic_stream_config(self):
        """Get dynamic stream configuration"""
        return self._config['dynamic_stream']
    
    def get_models_config(self):
        """Get models configuration"""
        return self._config['edge']['models']
    
    def get_queue_max_length(self):
        """Get maximum queue length for dynamic stream"""
        return self._config['queue']['max_length']
    
    def get_queue_low_threshold_length(self):
        """Get queue low length threshold for model switching"""
        return self._config['queue']['low_threshold_length']
    
    def get_queue_high_threshold_length(self):
        """Get queue high length threshold for model switching"""
        return self._config['queue']['high_threshold_length']

# Create a singleton instance
config = Config()