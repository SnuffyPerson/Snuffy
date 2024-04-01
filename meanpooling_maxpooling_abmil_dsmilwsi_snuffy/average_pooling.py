import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.in_size = in_size
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))
        
    def forward(self, feats):
        x = self.fc(feats)
        avg = torch.mean(x, dim=0) # avg vector in shape C
        return feats, x, avg

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.in_size = feature_size
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        avg = torch.mean(c, dim=0) # avg vector in shape C
        return feats.view(feats.shape[0], -1), c, avg

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, avg): # N x K
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        avg = avg.view(1, -1)
        A = avg.repeat(V.shape[0], 1) # use the average score of that class as the same attention everywhere in shape N x C
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
        
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        x = x.view(-1, self.i_classifier.in_size)
        feats, classes, avg = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, avg)
        
        return classes, prediction_bag, A, B, avg
