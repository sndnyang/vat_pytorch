from .fnn import FNN
from source import layers as L


class MLP(FNN):
    def __init__(self, layer_sizes, top_bn=True):
        self.linear_layers = []
        self.bn_layers = []
        self.act_layers = []
        self.params = []
        layers = zip(layer_sizes[:-1], layer_sizes[1:])
        for i, (m, n) in enumerate(layers):
            # l = L.Linear(size=(m, n), initial_W=init_linear((m, n)))
            l = L.Linear(size=(m, n))
            self.linear_layers.append(l)
            if top_bn:
                bn = L.BatchNormalization(size=(n))
                self.bn_layers.append(None)
                self.params += l.params + bn.params
            else:
                if i < len(layer_sizes) - 2:
                    bn = L.BatchNormalization(size=(n))
                    self.bn_layers.append(bn)
                    self.params += l.params + bn.params
                else:
                    self.bn_layers.append(None)
                    self.params += l.params
        for i in range(len(self.linear_layers) - 1):
            self.act_layers.append(L.relu)
        self.act_layers.append(L.softmax)

    def forward_for_finetuning_batch_stat(self, input):
        return self.forward(input, finetune=True)

    def forward_no_update_batch_stat(self, input, train=True):
        return self.forward(input, train, False)

    def forward(self, input, train=True, update_batch_stat=True, finetune=False):
        h = input
        for l, bn, act in zip(self.linear_layers, self.bn_layers, self.act_layers):
            h = l(h)
            if bn is not None:
                h = bn(h, train=train, update_batch_stat=update_batch_stat, finetune=finetune)
            h = act(h)
        return h
