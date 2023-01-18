
import torch.nn as nn
import torch.nn.functional as F
from expanded_gcn.layers import GraphConvolution, GraphConvolutionModified


class GCN(nn.Module):
    def __init__(self, nlayers, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.nlayers = nlayers
        self.gcs = nn.ModuleList([GraphConvolution(nfeat, nhid)])

        for i in range(nlayers-2):
            self.gcs.append(GraphConvolution(nhid, nhid))

        self.gcs.append(GraphConvolution(nhid, nclass))

        self.dropout = dropout

    def forward(self, x, adjs):
        x = F.relu(self.gcs[0](x, adjs))
        x = F.dropout(x, self.dropout, training=self.training)

        for i in range(self.nlayers-2):
            x = F.relu(self.gcs[i](x, adjs))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gcs[-1](x, adjs)

        return F.log_softmax(x, dim=1)


class GCNModified(nn.Module):
    def __init__(self, nlayers, nfeat, nhid, nclass, nneighbors, dropout):
        super(GCNModified, self).__init__()

        self.nlayers = nlayers
        self.gcs = nn.ModuleList([GraphConvolutionModified(nfeat, nhid, nneighbors)])

        for i in range(nlayers-2):
            self.gcs.append(GraphConvolutionModified(nhid, nhid, nneighbors))

        self.gcs.append(GraphConvolutionModified(nhid, nclass, nneighbors))
        self.dropout = dropout

    def forward(self, x, adjs):
        x = F.relu(self.gcs[0](x, adjs))
        x = F.dropout(x, self.dropout, training=self.training)

        for i in range(self.nlayers-2):
            x = F.relu(self.gcs[i](x, adjs))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.gcs[-1](x, adjs)

        return F.log_softmax(x, dim=1)