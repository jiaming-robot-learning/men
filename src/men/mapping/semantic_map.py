
from .base_map import BaseMap, MapArgs


class SemanticMapArgs(MapArgs):
    
    num_obj_categories: int     = 0        # for semantic mapping 

    
    
class SemanticMap2D(BaseMap):
    
    def __init__(self, **kwargs):
        
        self.args = SemanticMapArgs(**kwargs)
        args = self.args
        self.num_obj_categories = args.num_obj_categories
        
        super().__init__()
        
    def _get_total_map_channels(self):
        return 2 + self.num_obj_categories