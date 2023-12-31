from tensorflow import keras
import re
from yapps import runtime

class NetScanner(runtime.Scanner):
    patterns = [
        ('"\\\\]"', re.compile('\\]')),
        ('"\\\\["', re.compile('\\[')),
        ('"\\\\)"', re.compile('\\)')),
        ('"\\\\("', re.compile('\\(')),
        ('"\\\\}"', re.compile('\\}')),
        ('","', re.compile(',')),
        ('"\\\\{"', re.compile('\\{')),
        ('\\s+', re.compile('\\s+')),
        ('NUM', re.compile('[0-9]+')),
        ('CONV', re.compile('C')),
        ('POOL', re.compile('P')),
        ('SPLIT', re.compile('S')),
        ('FC', re.compile('FC')),
        ('DROP', re.compile('D')),
        ('GLOBALAVE', re.compile('GAP')),
        ('NIN', re.compile('NIN')),
        ('BATCHNORM', re.compile('BN')),
        ('SOFTMAX', re.compile('SM')),
    ]
    def __init__(self, str,*args,**kw):
        runtime.Scanner.__init__(self,None,{'\\s+':None,},str,*args,**kw)
        
class NetGenerating(runtime.Parser):
    Context = runtime.Context
    def layers(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'layers', [])
        _token = self._peek('CONV', 'NIN', 'GLOBALAVE', 'BATCHNORM', 'POOL', 'SPLIT', 'FC', 'DROP', 'SOFTMAX', context=_context)
        if _token == 'CONV':
            conv = self.conv(_context)
            return conv
        elif _token == 'NIN':
            nin = self.nin(_context)
            return nin
        elif _token == 'GLOBALAVE':
            gap = self.gap(_context)
            return gap
        elif _token == 'BATCHNORM':
            bn = self.bn(_context)
            return bn
        elif _token == 'POOL':
            pool = self.pool(_context)
            return pool
        elif _token == 'SPLIT':
            split = self.split(_context)
            return split
        elif _token == 'FC':
            fc = self.fc(_context)
            return fc
        elif _token == 'DROP':
            drop = self.drop(_context)
            return drop
        else: 
            softmax = self.softmax(_context)
            return softmax
    def conv(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'conv', [])
        CONV = self._scan('CONV', context=_context)
        result = ['conv']
        numlist = self.numlist(_context)
        return result + numlist

    def nin(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'nin', [])
        NIN = self._scan('NIN', context=_context)
        result = ['nin']
        numlist = self.numlist(_context)
        return result + numlist

    def gap(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'gap', [])
        GLOBALAVE = self._scan('GLOBALAVE', context=_context)
        result = ['gap']
        numlist = self.numlist(_context)
        return result + numlist

    def bn(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'bn', [])
        BATCHNORM = self._scan('BATCHNORM', context=_context)
        return ['bn']

    def pool(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'pool', [])
        POOL = self._scan('POOL', context=_context)
        result = ['pool']
        numlist = self.numlist(_context)
        return result + numlist

    def fc(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'fc', [])
        FC = self._scan('FC', context=_context)
        result = ['fc']
        numlist = self.numlist(_context)
        return result + numlist

    def drop(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'drop', [])
        DROP = self._scan('DROP', context=_context)
        result = ['dropout']
        numlist = self.numlist(_context)
        return result + numlist

    def softmax(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'softmax', [])
        SOFTMAX = self._scan('SOFTMAX', context=_context)
        result = ['softmax']
        numlist = self.numlist(_context)
        return result + numlist

    def split(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'split', [])
        SPLIT = self._scan('SPLIT', context=_context)
        self._scan('"\\\\{"', context=_context)
        result = ['split']
        net = self.net(_context)
        result.append(net)
        while self._peek('"\\\\}"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            net = self.net(_context)
            result.append(net)
        self._scan('"\\\\}"', context=_context)
        return result

    def numlist(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'numlist', [])
        self._scan('"\\\\("', context=_context)
        result = []
        NUM = self._scan('NUM', context=_context)
        result.append(int(NUM))
        while self._peek('"\\\\)"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            NUM = self._scan('NUM', context=_context)
            result.append(int(NUM))
        self._scan('"\\\\)"', context=_context)
        return result

    def net(self, _parent=None):
        _context = self.Context(_parent, self._scanner, 'net', [])
        self._scan('"\\\\["', context=_context)
        result = []
        layers = self.layers(_context)
        result.append(layers)
        while self._peek('"\\\\]"', '","', context=_context) == '","':
            self._scan('","', context=_context)
            layers = self.layers(_context)
            result.append(layers)
        self._scan('"\\\\]"', context=_context)
        return result

    
def parse(rule, text):
    P = NetGenerating(NetScanner(text))
    return runtime.wrap_error_reporter(P, rule)



def caffe_to_keras(layer, first = False):
    if layer[0] == 'conv':
        if first:
            return keras.layers.Conv2D(
            layer[1],
            layer[2],
            strides = (layer[3], layer[3]),
        )
        else:
            return keras.layers.Conv2D(
            layer[1],
            layer[2],
            strides = (layer[3], layer[3]),
            input_shape = (28,28,1)
        )
    elif layer[0] == "dropout":
        return keras.layers.Dropout(layer[1]/(layer[2] * 2))
    elif layer[0] == "fc":
        return keras.layers.Dense(
        units=layer[1],
    )
    elif layer[0] == "softmax":
         return keras.layers.Dense(
        10, activation = "softmax")
    elif layer[0]== "pool":
        return keras.layers.MaxPooling2D(
        pool_size = (layer[1], layer[1]), 
        strides = layer[2],
        )
    elif layer[0] == "gap":
        return keras.layers.GlobalAveragePooling2D()
    else:
        raise Exception

def parse_network_structure(net):
    keras.backend.clear_session()

    structure = []
    for layer_dict in net:
        new_layer =caffe_to_keras(layer_dict)
        if layer_dict[0] in ["softmax", "fc"]:
            structure.append(keras.layers.Flatten())
        structure.append(new_layer)
    return structure