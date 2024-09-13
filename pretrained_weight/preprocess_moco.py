import torch 

pretrained_net = "resnet50_inet_moco_v2_800ep.pth"

pretrained_net_dict = torch.load(pretrained_net)

new_state_dict = pretrained_net_dict['state_dict'].copy()
for key in pretrained_net_dict['state_dict'].keys() : 
    new_key = key.replace('module.encoder_q.', '')
    new_state_dict[new_key] = pretrained_net_dict['state_dict'][key]
    del new_state_dict[key]

torch.save(new_state_dict, 'resnet50_inet_moco_v2_800ep.pth')
