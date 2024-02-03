import timm

from torch import nn, Tensor
import torch

from transformer import Transformer

class PressleyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.image_embedding = ImageEncoder()
        self.action_embedding = ActionEncoder()
        self.action_decoder = ActionDecoder()
        self.transfomer_model = Transformer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, prev_camera, prev_actions, training: bool = False) -> Tensor:
        """
        Arguments:
            prev_camera: Tensor, shape ``[seq_len, batch_size]``
        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """

        reshaped_image = prev_camera.view(-1, 3, 224, 224).to(self.device)
        camera_tokens = self.image_embedding(reshaped_image).reshape(10, 5, 256)
        
        action_tokens = self.action_embedding(prev_actions.to(self.device))
        tokens = torch.tensor([]).to(self.device)
        
        tokens = torch.cat([camera_tokens, action_tokens], dim=2) #B T Emb

        # use normal transformer mask
        attention_mask = torch.tril(torch.ones(camera_tokens.shape[0], camera_tokens.shape[0])).to(self.device)

        # B T
        output, score = self.transfomer_model(tokens, training, attention_mask)
        output = self.action_decoder(output)
        return output

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0)
        self.fc = nn.Linear(2048, 256)
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        y1 = self.model(x)
        y2 = self.fc(y1) # make 256 size embeddings

        return y2
    
class ActionEncoder(nn.Module):
    
    def __init__ (self,  n_actions=7):
        super().__init__()
        # lets start with one layer and then make one NN for each axis
        self.lin = nn.Linear(n_actions, 256)

    def forward(self, x):
        return self.lin(x)
    
class ActionDecoder(nn.Module):
    
    def __init__ (self, n_actions=7):
        super().__init__()
        self.lin = nn.Linear(256, n_actions)

    def forward(self, x):
        return self.lin(x)