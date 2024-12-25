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
        # Get project root directory (parent of edge/cloud directories)
        project_root = Path(__file__).parent
        config_path = project_root / 'config.yaml'
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
            
        # Convert relative paths to absolute using project root
        # Video source path
        video_path = self._config['stream']['source_path']
        if not Path(video_path).is_absolute():
            self._config['stream']['source_path'] = str(project_root / video_path)
            
        # Model weights directory
        weights_dir = self._config['edge']['models']['weights_dir']
        if not Path(weights_dir).is_absolute():
            self._config['edge']['models']['weights_dir'] = str(project_root / weights_dir)
    
    def get_cloud_display_config(self):
        return self._config['cloud']['display_server']
    
    def get_edge_processing_config(self):
        return self._config['edge']['processing_server']
    
    def get_stream_config(self):
        return self._config['stream']
    
    def get_models_config(self):
        return self._config['edge']['models']

# Create a singleton instance
config = Config()