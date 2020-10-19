import torch
from torch.nn import Module, Parameter, init
from torch.nn.functional import relu


class Encoder(Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, #cuda=False, 
            feature_transform=False): 
        super().__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        #self.cuda = cuda
        #self.aggregator.cuda = cuda
        weight_dim_y = self.feat_dim if self.gcn else 2 * self.feat_dim
        self.weight = Parameter(torch.empty(embed_dim, weight_dim_y))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        neigh_feats = self.aggregator.forward(nodes, to_neighs, self.num_sample)
        if not self.gcn:
            combined = neigh_feats
#            if self.cuda:
#                self_feats = self.features(torch.LongTensor(nodes).cuda())
#            else:
#                self_feats = self.features(torch.LongTensor(nodes))
#            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            self_feats = self.features(torch.tensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = relu(self.weight.mm(combined.t()))
        return combined
