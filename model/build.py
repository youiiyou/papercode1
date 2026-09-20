from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task() #解析loss_names 决定启用哪些任务

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:#本质是分类任务
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)#正态分布，用到了随机值，即由随机种子影响的固定随机值
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim, #特征总维度
                                                    self.embed_dim // 64, #头数
                                                    batch_first=True) #输入输出张量以batch开头
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth, #框架图为4层
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5 #1/sqrt(emb_dim) 放在attention的分母
            
            #下列为Transformer基石，均值0，方差1归一化
            self.ln_pre_t = LayerNorm(self.embed_dim) #层归一化，   用于文本特征Q
            self.ln_pre_i = LayerNorm(self.embed_dim) #层归一化，   用于图像特征K,V
            self.ln_post = LayerNorm(self.embed_dim) #层归一化，   用于cross attention输出

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)#用网络深度初始化投影层
            attn_std = scale#初始化注意力层内部投影的标准差
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5#初始化前馈层中间层的标准差
            for block in self.cross_modal_transformer.resblocks:#对残差块做初始化
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)#1//——512
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn  独立的多头交叉注意力层
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential( #预测头 这是一个小型神经网络 有四层
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))])) #最终预测层
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self): #决定训练哪些任务
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')] #拆分成列表
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q), #这些都是层归一化
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0] #这个会返回一个元组，第一个元素是输出，第二个是注意力权重此处用[0]提取第一个
        x = x.permute(1, 0, 2)  # NLD -> LND  NLD:batch len dim
        x = self.cross_modal_transformer(x) #见上为Transformer
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):#本质是卷积
        x = self.base_model.encode_image(image) #x通常是(batch_size, seq_len, embed_dim)
        return x[:, 0, :].float()#batch取全部，序列取第0个位置，特征维度取全部。于是去到了所有的局部特征
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()#花式索引，每一个eos位置的全局文本特征

    def forward(self, batch):#前向传播：数据从输入到输出的过程，向前流经所有层，相对于反向传播而言。
        ret = dict()

        #1)特征提取
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float() #提取全局图像特征
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale#ke
        ret.update({'temperature': 1 / logit_scale})

        #2)3)调用任务头与计算损失
        #有的loss有参数需要更新，在init里面定义，有的loss无参数直接计算
        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)#此处得到1维张量，应仔细理解
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels) #找到所有非零元素的位置索引(未掩码位置标记为0)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc}) #acc正确比例

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model) #转换成fp16
    return model
