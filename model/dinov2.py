import torch
import torch.nn as nn
import torch.hub 



dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}

class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)





class DINOv2(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_b'):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', dino_backbones[backbone]['name'])
        embedding_size = dino_backbones[backbone]['embedding_size']
        self.head = nn.Linear(embedding_size, num_classes)
        

    def forward(self, x):
        x = self.backbone(x)
        ## self.backbone.get_intermediate_layers(x, n=3, reshape=True) 
        ## last 3 layers
        ## output: a length of 3 tuple, each is with torch.Size([bs, embed_dim, h, w])
        x = self.head(x)
        return x



def dinov2_small_patch14(nb_cls, **kwargs):
    model = DINOv2(nb_cls, backbone='dinov2_s')
    return model


def dinov2_base_patch14(nb_cls, **kwargs):
    model = DINOv2(nb_cls, backbone='dinov2_b')
    return model

if __name__ == "__main__":
    #model = dinov2_base_patch14(nb_cls = 100)
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.cuda()
    img_size = 378
    x = torch.cuda.FloatTensor(4, 3, img_size, img_size)
    with torch.no_grad() : 
        y = model.get_intermediate_layers(x, n=3, reshape=True) 
        print (y[0].shape)
        print (len(y))
    
    model = dinov2_base_patch14(nb_cls = 100)
    model.cuda()
    img_size = 378
    x = torch.cuda.FloatTensor(4, 3, img_size, img_size)
    with torch.no_grad() : 
        y = model(x) 
        print (y.shape)
        