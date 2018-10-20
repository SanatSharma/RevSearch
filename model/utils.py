# utils.py This file may be used for all utility functions

'''
 Create a bijection betweeen int and object. May be used for reverse indexing
'''
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.ints_to_objs)

    def contains(self, obj):
        return self.index_of(obj) != -1
    
    def index_of(self, obj):
        if obj in self.objs_to_ints:
            return self.objs_to_ints[obj]
        return -1
    
    # Get the index of the object, if add_object, add object to dict if not present
    def get_index(self, obj, add_object = True):
        if not add or obj in self.objs_to_ints:
            return self.index_of(obj)
        new_idx = len(self.ints_to_objs)
        self.objs_to_ints[obj] = new_idx
        self.ints_to_objs[new_idx] = obj
        return new_idx

           

