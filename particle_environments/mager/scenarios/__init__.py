import imp
import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    if not osp.exists(pathname):
        import multiagent.scenarios as scenario
        pathname = osp.join(osp.dirname(scenario.__file__), name)

    return imp.load_source('', pathname)
