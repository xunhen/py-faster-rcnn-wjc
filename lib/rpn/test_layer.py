import caffe
from fast_rcnn.config import cfg
from utils.timer import Timer
import yaml
import time
DEBUG = False

class TestLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._name = layer_params['name']
        pass

    def forward(self, bottom, top):

        top[0]=bottom[0]

        cfg.timer.toc()
        if cfg.BEGIN:
            last=cfg.time_ave.get(self._name,0)
            cfg.time_ave[self._name]=last+cfg.timer.diff*1000
        print(" detect time test_layer is {:.8f}ms----{}".format(cfg.timer.diff*1000,self._name))
        cfg.timer.tic()


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
