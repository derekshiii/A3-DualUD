import torch
import torch.nn as nn
import torch.nn.functional as F
import os,random
from einops import rearrange
from models import get_model
from dataloaders import MyDataset,PatientDataset,MyBatchSampler
from torch.utils.data import DataLoader
from losses import MultiClassDiceLoss,BoundarySensitiveDiceLoss

import numpy as np
from skimage import io

from utils import IterationCounter, Visualizer, mean_dice, SoftMatchWeighting,FixedThresholding, DistAlignEMA, mean_assd

from tqdm import tqdm

import pdb
import matplotlib.pyplot as plt


class PseudoLabel_Trainer():
    def __init__(self, opt):
        self.opt = opt
    
    def initialize(self):
        
        ### initialize dataloaders
        if self.opt['patient_level_dataloader']:
            train_dataset = PatientDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True, weak_strong_aug=True)
            patient_sampler = MyBatchSampler(train_dataset,self.opt['batch_size'])
            self.train_dataloader = DataLoader(train_dataset,batch_sampler=patient_sampler,num_workers=self.opt['num_workers'])
        else:
            self.train_dataloader = DataLoader(
                MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='train', split_train=True, weak_strong_aug=True),
                batch_size=self.opt['batch_size'],
                shuffle=True,
                drop_last=True,
                num_workers=self.opt['num_workers']
            )

        print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset(self.opt['data_root'], self.opt['target_sites'], phase='val', split_train=False),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models
        self.use_ema = self.opt['use_ema']
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        model_state_dict = checkpoint['model']
        if 'classifier.final_layer.0.weight' in model_state_dict:
            model_state_dict['classifier.final_layer.weight'] = model_state_dict.pop('classifier.final_layer.0.weight')
        self.model = get_model(self.opt)
        self.model.load_state_dict(model_state_dict)


        # self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.opt['gpu_id'])
        
        if self.use_ema:
            self.ema_model = get_model(self.opt)
            # self.ema_model.load_state_dict(checkpoint)
            self.ema_model.load_state_dict(model_state_dict)
            self.ema_model = self.ema_model.to(self.opt['gpu_id'])
            self.ema_model.eval()
        
        self.total_epochs = self.opt['total_epochs']
        self.total_steps = self.total_epochs * len(self.train_dataloader)
       
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        ## pseudo label setting
        self.match_type = self.opt['match_type']
        
        
        if self.match_type == 'naive':
            self.masking = None
        elif self.match_type == 'fixmatch':
            self.masking = FixedThresholding(self.opt['p_cutoff'])
        elif self.match_type == 'softmatch':
            self.masking = SoftMatchWeighting(self.opt['num_classes'],per_class=self.opt['per_class'])
            self.use_dist_align = self.opt['use_dist_align']
            if self.use_dist_align:
                self.dist_align = DistAlignEMA(self.opt['num_classes'])
        ## losses
        self.criterion_pseudo = nn.CrossEntropyLoss(weight=torch.tensor([0.1,1,1,1,1]).to(self.opt['gpu_id']),reduction='none')
        self.criterian_dc = MultiClassDiceLoss(self.opt)
        self.criterian_bd = BoundarySensitiveDiceLoss(self.opt)
        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.set_seed(self.opt['random_seed'])
        self.model_resume()
        
    def set_seed(self,seed):     
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    def save_models(self, step, dice):
        if step != 0:
            checkpoint_dir = self.opt['checkpoint_dir']
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
            torch.save(state, os.path.join(checkpoint_dir, 'saved_models', 'model_step_{}_dice_{:.4f}.pth'.format(step,dice)))

    
    def save_best_models(self, step, dice):
        checkpoint_dir = self.opt['checkpoint_dir']
        for file in os.listdir(os.path.join(checkpoint_dir, 'saved_models')):
            if 'best_model' in file:
                os.remove(os.path.join(checkpoint_dir, 'saved_models', file))
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        torch.save(state,os.path.join(checkpoint_dir, 'saved_models','best_model_step_{}_dice_{:.4f}.pth'.format(step,dice)))


    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params,lr=self.opt['lr'],betas=(0.9, 0.999), weight_decay=0.0005)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return optimizer, scheduler
    
    def model_resume(self):
        if self.opt['continue_train']:
            if os.path.isfile(self.opt['resume']):
                print("=> Loading checkpoint '{}'".format(self.opt['resume']))
            state = torch.load(self.opt['resume'])
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.start_epoch = state['epoch']
        else:
            self.start_epoch = 0
            print("=> No checkpoint, train from scratch !")
    
    def ema_update(self):
        # Use the true average until the exponential average is more correct
        global_step = self.iter_counter.steps_so_far
        ema_decay = self.opt['ema_decay']
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(ema_decay).add_(param.data*(1 - ema_decay))

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()
        
        # get losses
        self.model.train()
        self.ema_model.train()
        imgs_w, imgs_s, gt = data[0],data[1],data[2]
        b, c, h, w = imgs_w.shape
        _, prob_s = self.model(imgs_s)


        pred_s = torch.argmax(prob_s, dim=1)
        # prob_s = rearrange(prob_s, 'b c h w -> (b h w) c')
        # model是presourcemodel，emamodel是目标模型
        
        with torch.no_grad():
            if self.use_ema:
                _, prob_w = self.ema_model(imgs_w)
                # -----------------------------------------------------
                # pseudo——label
                # uncertain method

                preds = torch.zeros([10, imgs_w.shape[0], 5, imgs_w.shape[2], imgs_w.shape[3]]).cuda()
                features = torch.zeros([10, imgs_w.shape[0], 256, 64, 64]).cuda()
                for i in range(10):
                    with torch.no_grad():
                        features[i, ...], preds[i, ...] = self.ema_model(imgs_s)
                        # _, preds[i, ...] = self.ema_model(imgs_s)

                preds = torch.sigmoid(preds / 2.0)
                std_map = torch.std(preds, dim=0)

                # ###############################################################
                # 假设我们要可视化第一个样本第一个通道的标准差图
                std_map_all_channels = std_map[0, :, :, :].cpu().numpy()

                # 对通道进行平均
                fused_map = np.mean(std_map_all_channels, axis=0)
                plt.imshow(fused_map, cmap='coolwarm')
                # plt.imshow(std_map[0, 1, :, :].cpu().numpy(), cmap='gray')  # 使用 'hot' 或 'gray' 作为颜色映射
                plt.colorbar()

                # 设置标题和坐标轴的字体大小
                plt.title("Uncertain Map", fontsize=16)  # 调整标题字体大小
                plt.xlabel("X-axis", fontsize=14)  # 设置 X 轴标签和字体大小
                plt.ylabel("Y-axis", fontsize=14)  # 设置 Y 轴标签和字体大小

                # 调整刻度标签的字体大小
                plt.xticks(fontsize=12)  # X 轴刻度字体大小
                plt.yticks(fontsize=12)  # Y 轴刻度字体大小
                plt.show()
                # plt.imshow(std_map[0, 4, :, :].cpu().numpy(), cmap='coolwarm')  # 使用 'hot' 或 'gray' 作为颜色映射
                # plt.colorbar()
                # plt.title("Uncertain Map")
                # plt.show()
                # std_map =preds求方差
                # 将 std_map 张量调整到与 feature 张量相同的空间尺寸

                mask_0_obj = torch.zeros(
                    [std_map.shape[0], 1, std_map.shape[2], std_map.shape[3]]).cuda()
                mask_1_obj = torch.zeros(
                    [std_map.shape[0], 1, std_map.shape[2], std_map.shape[3]]).cuda()
                mask_2_obj = torch.zeros(
                    [std_map.shape[0], 1, std_map.shape[2], std_map.shape[3]]).cuda()
                mask_3_obj = torch.zeros(
                    [std_map.shape[0], 1, std_map.shape[2], std_map.shape[3]]).cuda()
                mask_4_obj = torch.zeros(
                    [std_map.shape[0], 1, std_map.shape[2], std_map.shape[3]]).cuda()
                mask_0_obj[std_map[:, 0:1, ...] < 0.01] = 1.0
                mask_1_obj[std_map[:, 1:2, ...] < 0.01] = 1.0
                mask_2_obj[std_map[:, 2:3, ...] < 0.01] = 1.0
                mask_3_obj[std_map[:, 3:4, ...] < 0.01] = 1.0
                mask_4_obj[std_map[:, 4:, ...] < 0.01] = 1.0

                mask_0 = mask_0_obj
                mask_1 = mask_1_obj
                mask_2 = mask_2_obj
                mask_3 = mask_3_obj
                mask_4 = mask_4_obj
                # mask[16,5,64,64]
                mask1 = torch.cat((mask_0, mask_1, mask_2, mask_3, mask_4), dim=1)

            # ------------------------------------------------------------------------------------
            #     preds = torch.zeros([10, imgs_w.shape[0], 5, imgs_w.shape[2], imgs_w.shape[3]]).cuda()
            #     features = torch.zeros([10, imgs_w.shape[0], 256, 64, 64]).cuda()
            #     for i in range(10):
            #         with torch.no_grad():
            #             features[i, ...], preds[i, ...] = self.ema_model(imgs_s)
            #     preds1 = torch.sigmoid(preds)
            #     prediction = torch.mean(preds1, dim=0)
            #     prediction_yuan = torch.mean(preds, dim=0)
            #     # prediction=用preds1求均值
            #     preds = torch.sigmoid(preds / 2.0)
            #     std_map = torch.std(preds, dim=0)
            #     # std_map =preds求方差
            #     feature = torch.mean(features, dim=0)
            #     pseudo_label = prediction.clone()
            #     pseudo_label[pseudo_label > 0.8] = 1.0
            #     pseudo_label[pseudo_label <= 0.8] = 0.0
            #
            #     target_0_obj = F.interpolate(pseudo_label[:, 0:1, ...], size=feature.size()[2:], mode='nearest')
            #     target_1_obj = F.interpolate(pseudo_label[:, 1:2, ...], size=feature.size()[2:], mode='nearest')
            #     target_2_obj = F.interpolate(pseudo_label[:, 2:3, ...], size=feature.size()[2:], mode='nearest')
            #     target_3_obj = F.interpolate(pseudo_label[:, 3:4, ...], size=feature.size()[2:], mode='nearest')
            #     target_4_obj = F.interpolate(pseudo_label[:, 4:, ...], size=feature.size()[2:], mode='nearest')
            #     target_0_bck = 1.0 - target_0_obj
            #     target_1_bck = 1.0 - target_1_obj
            #     target_2_bck = 1.0 - target_2_obj
            #     target_3_bck = 1.0 - target_3_obj
            #     target_4_bck = 1.0 - target_4_obj
            #     prediction_small = F.interpolate(prediction, size=feature.size()[2:], mode='bilinear',
            #                                      align_corners=True)
            #     std_map_small = F.interpolate(std_map, size=feature.size()[2:], mode='bilinear', align_corners=True)
            #
            #     mask_0_obj = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_0_bck = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_1_obj = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_1_bck = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_2_obj = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_2_bck = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_3_obj = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_3_bck = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_4_obj = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_4_bck = torch.zeros(
            #         [std_map_small.shape[0], 1, std_map_small.shape[2], std_map_small.shape[3]]).cuda()
            #     mask_0_obj[std_map_small[:, 0:1, ...] < 0.05] = 1.0
            #     mask_0_bck[std_map_small[:, 0:1, ...] < 0.05] = 1.0
            #     mask_1_obj[std_map_small[:, 1:2, ...] < 0.05] = 1.0
            #     mask_1_bck[std_map_small[:, 1:2, ...] < 0.05] = 1.0
            #     mask_2_obj[std_map_small[:, 2:3, ...] < 0.05] = 1.0
            #     mask_2_bck[std_map_small[:, 2:3, ...] < 0.05] = 1.0
            #     mask_3_obj[std_map_small[:, 3:4, ...] < 0.05] = 1.0
            #     mask_3_bck[std_map_small[:, 3:4, ...] < 0.05] = 1.0
            #     mask_4_obj[std_map_small[:, 4:, ...] < 0.05] = 1.0
            #     mask_4_bck[std_map_small[:, 4:, ...] < 0.05] = 1.0
            #     mask_0 = mask_0_obj+mask_0_bck
            #     mask_1 = mask_1_obj+mask_1_bck
            #     mask_2 = mask_2_obj+mask_2_bck
            #     mask_3 = mask_3_obj+mask_3_bck
            #     mask_4 = mask_4_obj+mask_4_bck
            #     mask1 = torch.cat((mask_0, mask_1, mask_2, mask_3, mask_4), dim=1)
            #
            #
            #     feature_0_obj = feature * target_0_obj * mask_0_obj
            #     feature_1_obj = feature * target_1_obj * mask_1_obj
            #     feature_2_obj = feature * target_2_obj * mask_2_obj
            #     feature_3_obj = feature * target_3_obj * mask_3_obj
            #     feature_0_bck = feature * target_0_bck * mask_0_bck
            #     feature_1_bck = feature * target_1_bck * mask_1_bck
            #     feature_2_bck = feature * target_2_bck * mask_2_bck
            #     feature_3_bck = feature * target_3_bck * mask_3_bck
            #     feature_4_obj = feature * target_4_obj * mask_4_obj
            #
            #     feature_4_bck = feature * target_4_bck*mask_4_bck
            #
            #     centroid_0_obj = torch.sum(feature_0_obj * prediction_small[:, 0:1, ...], dim=[0, 2, 3], keepdim=True)
            #     centroid_1_obj = torch.sum(feature_1_obj * prediction_small[:, 1:2, ...], dim=[0, 2, 3], keepdim=True)
            #     centroid_2_obj = torch.sum(feature_2_obj * prediction_small[:, 2:3, ...], dim=[0, 2, 3], keepdim=True)
            #     centroid_3_obj = torch.sum(feature_3_obj * prediction_small[:, 3:4, ...], dim=[0, 2, 3], keepdim=True)
            #     centroid_4_obj = torch.sum(feature_4_obj * prediction_small[:, 4:, ...], dim=[0, 2, 3], keepdim=True)
            #     centroid_0_bck = torch.sum(feature_0_bck * (1.0 - prediction_small[:, 0:1, ...]), dim=[0, 2, 3],
            #                                keepdim=True)
            #     centroid_1_bck = torch.sum(feature_1_bck * (1.0 - prediction_small[:, 1:2, ...]), dim=[0, 2, 3],
            #                                keepdim=True)
            #     centroid_2_bck = torch.sum(feature_2_bck * (1.0 - prediction_small[:, 2:3, ...]), dim=[0, 2, 3],
            #                                keepdim=True)
            #     centroid_3_bck = torch.sum(feature_3_bck * (1.0 - prediction_small[:, 3:4, ...]), dim=[0, 2, 3],
            #                                keepdim=True)
            #     centroid_4_bck = torch.sum(feature_4_bck * (1.0 - prediction_small[:, 4:, ...]), dim=[0, 2, 3],
            #                                keepdim=True)
            #
            #     target_0_obj_cnt = torch.sum(mask_0_obj *target_0_obj * prediction_small[:, 0:1, ...], dim=[0, 2, 3],
            #                                  keepdim=True)
            #     target_1_obj_cnt = torch.sum( mask_1_obj *target_1_obj * prediction_small[:, 1:2, ...], dim=[0, 2, 3],
            #                                  keepdim=True)
            #     target_2_obj_cnt = torch.sum( mask_2_obj *prediction_small[:, 2:3, ...], dim=[0, 2, 3],
            #                                  keepdim=True)
            #     target_3_obj_cnt = torch.sum( mask_3_obj *target_3_obj * prediction_small[:, 3:4, ...], dim=[0, 2, 3],
            #                                  keepdim=True)
            #     target_4_obj_cnt = torch.sum(mask_4_obj *target_4_obj * prediction_small[:, 4:, ...], dim=[0, 2, 3],
            #                                  keepdim=True)
            #     target_0_bck_cnt = torch.sum(mask_0_bck *target_0_bck * (1.0 - prediction_small[:, 0:1, ...]),
            #                                  dim=[0, 2, 3], keepdim=True)
            #     target_1_bck_cnt = torch.sum(mask_1_bck *target_1_bck * (1.0 - prediction_small[:, 1:2, ...]),
            #                                  dim=[0, 2, 3], keepdim=True)
            #     target_2_bck_cnt = torch.sum(mask_2_bck * target_2_bck * (1.0 - prediction_small[:, 2:3, ...]),
            #                                  dim=[0, 2, 3], keepdim=True)
            #     target_3_bck_cnt = torch.sum( mask_3_bck *target_3_bck * (1.0 - prediction_small[:, 3:4, ...]),
            #                                  dim=[0, 2, 3], keepdim=True)
            #     target_4_bck_cnt = torch.sum( mask_4_bck *target_4_bck * (1.0 - prediction_small[:, 4:, ...]),
            #                                  dim=[0, 2, 3], keepdim=True)
            #
            #     centroid_0_obj /= target_0_obj_cnt
            #     centroid_1_obj /= target_1_obj_cnt
            #     centroid_2_obj /= target_2_obj_cnt
            #     centroid_3_obj /= target_3_obj_cnt
            #     centroid_4_obj /= target_4_obj_cnt
            #     centroid_0_bck /= target_0_bck_cnt
            #     centroid_1_bck /= target_1_bck_cnt
            #     centroid_2_bck /= target_2_bck_cnt
            #     centroid_3_bck /= target_3_bck_cnt
            #     centroid_4_bck /= target_4_bck_cnt
            #
            #     distance_0_obj = torch.sum(torch.pow(feature - centroid_0_obj, 2), dim=1, keepdim=True)
            #     distance_0_bck = torch.sum(torch.pow(feature - centroid_0_bck, 2), dim=1, keepdim=True)
            #     distance_1_obj = torch.sum(torch.pow(feature - centroid_1_obj, 2), dim=1, keepdim=True)
            #     distance_1_bck = torch.sum(torch.pow(feature - centroid_1_bck, 2), dim=1, keepdim=True)
            #     distance_2_obj = torch.sum(torch.pow(feature - centroid_2_obj, 2), dim=1, keepdim=True)
            #     distance_2_bck = torch.sum(torch.pow(feature - centroid_2_bck, 2), dim=1, keepdim=True)
            #     distance_3_obj = torch.sum(torch.pow(feature - centroid_3_obj, 2), dim=1, keepdim=True)
            #     distance_3_bck = torch.sum(torch.pow(feature - centroid_3_bck, 2), dim=1, keepdim=True)
            #     distance_4_obj = torch.sum(torch.pow(feature - centroid_4_obj, 2), dim=1, keepdim=True)
            #     distance_4_bck = torch.sum(torch.pow(feature - centroid_4_bck, 2), dim=1, keepdim=True)
            #
            #     proto_pseudo_0 = torch.zeros([data[1].shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            #     proto_pseudo_1 = torch.zeros([data[1].shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            #     proto_pseudo_2 = torch.zeros([data[1].shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            #     proto_pseudo_3 = torch.zeros([data[1].shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            #     proto_pseudo_4 = torch.zeros([data[1].shape[0], 1, feature.shape[2], feature.shape[3]]).cuda()
            #     proto_pseudo_0[distance_0_obj < distance_0_bck] = 1.0
            #     proto_pseudo_1[distance_1_obj < distance_1_bck] = 1.0
            #     proto_pseudo_2[distance_2_obj < distance_2_bck] = 1.0
            #     proto_pseudo_3[distance_3_obj < distance_3_bck] = 1.0
            #     proto_pseudo_4[distance_4_obj < distance_4_bck] = 1.0
            #
            #     proto_pseudo = torch.cat(
            #         (proto_pseudo_0, proto_pseudo_1, proto_pseudo_2, proto_pseudo_3, proto_pseudo_4), dim=1)
            #     proto_pseudo = F.interpolate(proto_pseudo, size=data[1].size()[2:], mode='nearest')
                # proto_pseudo = F.interpolate(mask1, size=data[1].size()[2:], mode='nearest')

                # prob_w = prediction_yuan * mask
            else:
                self.model.eval()
                _, prob_w = self.model(imgs_w)
                self.model.train()
        prob_w = prob_w*mask1
        # prob_w = rearrange(prob_w, 'b c h w -> (b h w) c')
        prob_w = F.softmax(prob_w,dim=1)

        pseudo_label = torch.argmax(prob_w, dim=1)
        if self.match_type == 'naive':
            mask = torch.ones_like(pseudo_label).float()
        elif self.match_type == 'fixmatch':
            mask = self.masking.masking(prob_w)
        elif self.match_type == 'softmatch':
            if self.use_dist_align:
                prob_w = self.dist_align.dist_align(prob_w)
            mask = self.masking.masking(prob_w)
        
        mask = mask.to(prob_s.device)
        pseudo_ce_loss = self.criterion_pseudo(prob_s, pseudo_label)
        # pseudo_dc_loss = self.criterian_dc(prob_s, pseudo_label)
        # pseudo_bd_loss = self.criterian_bd(prob_s, pseudo_label)
        pseudo_ce_loss = pseudo_ce_loss * mask
        # pseudo_dc_loss = pseudo_dc_loss * mask
        # pseudo_bd_loss = pseudo_bd_loss * mask

        # loss = pseudo_ce_loss.mean()+pseudo_dc_loss.mean()
        loss = pseudo_ce_loss.mean()
        # loss = pseudo_bd_loss.mean()
        
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        gt = gt.flatten()
        fg_idxs = gt > 0
        # fg_quantity = mask[fg_idxs].mean()
        # tp_region = (pseudo_label[fg_idxs] == gt[fg_idxs]) * (mask[fg_idxs] / torch.sum(mask[fg_idxs]))
        # fg_quality  = torch.sum(tp_region)
        trade_off = {}
        adapt_losses = {}
        # adapt_losses['pseudo_ce_loss'] = loss.detach()
        # trade_off['quality'] = fg_quality.detach()
        # trade_off['quantity'] = fg_quantity.detach()
        pseudo_label[mask == 0] = 5
        # pseudo_label = rearrange(pseudo_label,'(b h w) -> b h w',b=b,h=h,w=w)
        return adapt_losses,trade_off,pred_s,pseudo_label
    
    @torch.no_grad()
    def validate_one_step(self, data):
        
        self.model.eval()
        imgs = data[0]
        h,w = imgs.shape[2:]
        _,pred = self.model(imgs)
        pred = F.softmax(pred, dim=1)
        return pred

    
    def launch(self):
        self.initialize()
        self.train()

    def train(self):
        for epoch in range(self.start_epoch,self.total_epochs):
            train_iterator = tqdm((self.train_dataloader), total = len(self.train_dataloader))
            train_losses = {}
            for it, (img_s,img_w,segs,_) in enumerate(train_iterator):
                # pdb.set_trace()
                img_s = img_s.to(self.opt['gpu_id'])
                img_w = img_w.to(self.opt['gpu_id'])
                segs = segs.to(self.opt['gpu_id'])

                with self.iter_counter.time_measurement("train"):
                    losses,trade_off,pred_s,pred_w = self.train_one_step([img_s,img_w,segs])
                    if self.use_ema:
                        self.ema_update()
                    for k,v in losses.items():
                        train_losses[k] = v + train_losses.get(k,0)
                    train_iterator.set_description(f'Train Epoch [{epoch}/{self.total_epochs}]')
                    # train_iterator.set_postfix(ce_loss = train_losses['pseudo_ce_loss'].item()/(it+1), quality = trade_off['quality'].item(), quantity = trade_off['quantity'].item())

                with self.iter_counter.time_measurement("maintenance"):
                    if self.iter_counter.needs_displaying():
                        visuals = {'images':img_s[:,1].detach().cpu().numpy(),  'pred_s':pred_s.detach().cpu().numpy(),
                        'pred_w':pred_w.detach().cpu().numpy(),
                        'gt_segs':segs.detach().cpu().numpy()}
                        self.visualizer.display_current_Pseudo(self.iter_counter.steps_so_far,visuals)
                    self.visualizer.plot_current_losses(self.iter_counter.steps_so_far, losses)
                    self.visualizer.plot_current_metrics(self.iter_counter.steps_so_far,trade_off,'Quantity_vs_Quality')
                    if self.iter_counter.needs_evaluation_steps():
                        val_metrics = {}
                        sample_dict = {}
                        val_metrics_std = {}
                        val_assd = {}
                        val_assd_std = {}
                        val_iterator = tqdm((self.val_dataloader), total = len(self.val_dataloader))
                        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

                            val_imgs = val_imgs.to(self.opt['gpu_id'])
                            val_segs = val_segs.to(self.opt['gpu_id'])

                            predict = self.validate_one_step([val_imgs, val_segs])
                            for i,name in enumerate(val_names):

                                sample_name,index = name.split('_')[0],int(name.split('_')[1])
                                sample_dict[sample_name] = sample_dict.get(sample_name,[]) + [(predict[i].detach().cpu(),val_segs[i].detach().cpu(),index)]

                        pred_results_list = []
                        gt_segs_list = []

                        for k in sample_dict.keys():

                            sample_dict[k].sort(key=lambda ele: ele[2])
                            preds = []
                            targets = []
                            for pred,target,_ in sample_dict[k]:
                                if target.sum()==0:
                                    continue
                                preds.append(pred)
                                targets.append(target)
                            pred_results_list.append(torch.stack(preds,dim=-1))
                            gt_segs_list.append(torch.stack(targets,dim=-1))

                        # pdb.set_trace()
                        val_metrics['dice'],val_metrics_std['dice_std'] = mean_dice(pred_results_list,gt_segs_list,self.opt['num_classes'],self.opt['organ_list'])
                        print(val_metrics)
                        print(val_metrics_std)
                        # val_assd['assd'], val_assd_std['assd_std'] = mean_assd(pred_results_list, gt_segs_list,
                        #                                                        self.opt['num_classes'],
                        #                                                        self.opt['organ_list'])
                        # print(val_assd)
                        # print(val_assd_std)

                        if val_metrics['dice']['dice_avg'] > self.best_avg_dice:
                            self.best_avg_dice = val_metrics['dice']['dice_avg']
                            self.save_best_models(self.iter_counter.steps_so_far,val_metrics['dice']['dice_avg'])
                        else:
                            if self.iter_counter.needs_saving_steps():
                                self.save_models(self.iter_counter.steps_so_far,val_metrics['dice']['dice_avg'])
                        self.visualizer.plot_current_metrics(self.iter_counter.steps_so_far, val_metrics['dice'],'Dice_metrics')
                        self.schedular.step()
                self.iter_counter.record_one_iteration()
            self.iter_counter.record_one_epoch()

        # best_model_path = '/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/target_adapt/DeepLab_Abdomen_MR2CT_Adapt_FixMatch_fold4/exp_8_time_2024-09-20 17-51-16onlyp/saved_models/best_model_step_140_dice_0.8080.pth'
        # checkpoint = torch.load(best_model_path)
        # self.model.load_state_dict(checkpoint['model'])
        # self.model.eval()
        #
        # with torch.no_grad():
        #     val_iterator = tqdm((self.val_dataloader), total=len(self.val_dataloader))
        #     for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):
        #
        #         val_imgs = val_imgs.to(self.opt['gpu_id'])
        #         val_segs = val_segs.to(self.opt['gpu_id'])
        #
        #         predict = self.validate_one_step([val_imgs, val_segs])
        #
        #         output = torch.argmax(predict, dim=1)
        #         output_ = output.cpu().numpy()
        #         gt_output = val_segs.cpu().numpy()
        #         img_output = val_imgs.cpu().numpy()
        #
        #
        #         results = "/media/dell/disk2/zhoujie/MICCAI23-deeplab_TCT/pvalue/onlypseudo"
        #         if not os.path.exists(results):
        #             os.makedirs(results)
        #         # results_gt = "prediction_gt_fold4_1"
        #
        #
        #         # if not os.path.exists(results_gt):
        #         #     os.makedirs(results_gt)
        #         for i in range(len(output_)):
        #             result = np.zeros((256, 256), dtype=np.uint8)
        #             # result_gt = np.zeros((256, 256), dtype=np.uint8)
        #
        #
        #             result[output_[i] == 1] = 50
        #             result[output_[i] == 2] = 100
        #             result[output_[i] == 3] = 150
        #             result[output_[i] == 4] = 200
        #             # result_gt[gt_output[i] == 1] = 50
        #             # result_gt[gt_output[i] == 2] = 100
        #             # result_gt[gt_output[i] == 3] = 150
        #             # result_gt[gt_output[i] == 4] = 200
        #
        #
        #             predict_dir = os.path.join(results, val_names[i] + ".png")
        #             # gt_dir = os.path.join(results_gt, val_names[i] + ".png")
        #
        #             io.imsave(predict_dir, result)
        #             # io.imsave(gt_dir, result_gt)



