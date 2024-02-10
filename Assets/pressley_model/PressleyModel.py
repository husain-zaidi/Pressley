import timm

from torch import nn, Tensor
import torch

from transformer import Transformer

class PressleyModel(nn.Module):

    def __init__(self, nmbed):
        super().__init__()

        self.image_embedding = ImageEncoder(nmbed)
        self.action_embedding = ActionEncoder(nmbed)
        self.action_decoder = ActionDecoder(nmbed)
        self.transfomer_model = Transformer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nmbed = nmbed

    def forward(self, prev_camera, prev_actions) -> Tensor:
        """
        Arguments:
            prev_camera: Tensor, shape ``[batch_size, seq_len, 3, 224, 224]``
            prev_actions: Tensor, shape ``[batch_size, seq_len, nmbed]``
        Returns:
            output Tensor of shape ``[batch_size, seq_len, action]``
        """

        batch_size = prev_camera.shape[0]
        seq_len = prev_camera.shape[1]

        reshaped_image = prev_camera.view(-1, 3, 224, 224).to(self.device)
        camera_tokens = self.image_embedding(reshaped_image).reshape(batch_size, seq_len, self.nmbed)
        
        action_tokens = self.action_embedding(prev_actions.to(self.device))
        tokens = torch.tensor([]).to(self.device)
        
        tokens = torch.cat([camera_tokens, action_tokens], dim=2) #B T Emb

        # use normal transformer mask
        attention_mask = torch.tril(torch.ones(camera_tokens.shape[0], camera_tokens.shape[0])).to(self.device)

        # B T
        output, score = self.transfomer_model(tokens, attention_mask)
        output = self.action_decoder(output)
        return output

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self, nmbed, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0)
        self.fc = nn.Linear(2048, nmbed)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        y1 = self.model(x)
        y2 = self.fc(y1) # make nmbed size embeddings

        return y2
    
class ActionEncoder(nn.Module):
    
    def __init__ (self, nmbed, n_actions=7):
        super().__init__()
        # lets start with one layer and then make one NN for each axis
        self.lin = nn.Linear(n_actions, nmbed)

    def forward(self, x):
        return self.lin(x)
    
class ActionDecoder(nn.Module):
    
    def __init__ (self, nmbed, n_actions=7):
        super().__init__()
        self.lin = nn.Linear(nmbed, n_actions)

    def forward(self, x):
        return self.lin(x)