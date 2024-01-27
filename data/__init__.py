import os.path as osp

def get_relative_path(path):
    return osp.join(osp.dirname(path).split("/")[-1], osp.basename(path))