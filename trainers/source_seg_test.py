import torch
import os, random
from models import get_model

from dataloaders import MyDataset
from torch.utils.data import DataLoader
from losses import MultiClassDiceLoss, PixelPrototypeCELoss

import numpy as np

from utils import IterationCounter, Visualizer, mean_dice, mean_assd

from tqdm import tqdm
import pdb


class SourceDomainTest():
    def __init__(self, opt):
        self.opt = opt

    def initialize(self):

        ### initialize dataloaders
        # self.train_dataloader = DataLoader(
        #     MyDataset(self.opt['data_root'], self.opt['source_sites'], phase='train', split_train=True),
        #     batch_size=self.opt['batch_size'],
        #     shuffle=True,
        #     drop_last=True,
        #     num_workers=self.opt['num_workers']
        # )
        #
        # print('Length of training dataset: ', len(self.train_dataloader))

        self.val_dataloader = DataLoader(
            MyDataset(self.opt['data_root'], self.opt['source_sites'], phase='test', split_train=False),
            batch_size=self.opt['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=4
        )

        print('Length of validation dataset: ', len(self.val_dataloader))

        ## initialize the models

        self.model = get_model(self.opt)
        checkpoint = torch.load(self.opt['source_model_path'],map_location='cpu')
        model_state_dict = checkpoint['model']
        self.model.load_state_dict(model_state_dict)

        self.model = self.model.to(self.opt['gpu_id'])
        self.total_epochs = self.opt['total_epochs']

        ## optimizers, schedulars
        self.optimizer, self.schedular = self.get_optimizers()
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

        ## losses
        self.criterian_pce = PixelPrototypeCELoss(self.opt)
        self.criterian_dc = MultiClassDiceLoss(self.opt)

        ## metrics
        self.best_avg_dice = 0

        # visualizations
        self.iter_counter = IterationCounter(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.set_seed(self.opt['random_seed'])
        self.model_resume()

    def set_seed(self, seed):

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print('Random seed for this experiment is {} !'.format(seed))

    def save_models(self, epoch, dice):
        if epoch != 0:
            checkpoint_dir = self.opt['checkpoint_dir']
            state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, os.path.join(checkpoint_dir, 'saved_models',
                                           'model_epoch_{}_dice_{:.4f}.pth'.format(epoch, dice)))

    def save_best_models(self, epoch, dice):
        checkpoint_dir = self.opt['checkpoint_dir']
        for file in os.listdir(os.path.join(checkpoint_dir, 'saved_models')):
            if 'best_model' in file:
                os.remove(os.path.join(checkpoint_dir, 'saved_models', file))
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(checkpoint_dir, 'saved_models',
                                       'best_model_epoch_{}_dice_{:.4f}.pth'.format(epoch, dice)))

    def get_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.opt['lr'], betas=(0.9, 0.999), weight_decay=0.0005)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.1, min_lr=1e-7) # maximize dice score
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)
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

    ###################### training logic ################################
    def train_one_step(self, data):
        # zero out previous grads
        self.optimizer.zero_grad()

        # get losses
        imgs = data[0]
        segs = data[1]

        _, predict = self.model(imgs)

        loss_pce = self.criterian_pce(predict, segs)
        loss_dc = self.criterian_dc(predict, segs)

        loss = loss_pce + loss_dc

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        seg_losses = {}
        seg_losses['train_dc'] = loss_dc.detach()
        seg_losses['train_ce'] = loss_pce.detach()
        seg_losses['train_total'] = loss.detach()

        return predict, seg_losses

    @torch.no_grad()
    def validate_one_step(self, data):
        self.model.eval()

        imgs = data[0]



        _, predict = self.model(imgs)


        self.model.train()

        return predict

    def launch(self):
        self.initialize()
        self.train()

    def train(self):

        val_metrics = {}
        sample_dict = {}
        val_metrics_std = {}
        val_assd = {}
        val_assd_std = {}

        val_iterator = tqdm((self.val_dataloader), total=len(self.val_dataloader))
        for it, (val_imgs, val_segs, val_names) in enumerate(val_iterator):

            val_imgs = val_imgs.to(self.opt['gpu_id'])
            val_segs = val_segs.to(self.opt['gpu_id'])

            predict = self.validate_one_step([val_imgs, val_segs])

            for i, name in enumerate(val_names):
                sample_name, index = name.split('_')[0], int(name.split('_')[1])
                sample_dict[sample_name] = sample_dict.get(sample_name, []) + [
                    (predict[i].detach().cpu(), val_segs[i].detach().cpu(), index)]


        pred_results_list = []
        gt_segs_list = []
        for k in sample_dict.keys():

            sample_dict[k].sort(key=lambda ele: ele[2])
            preds = []
            targets = []
            for pred, target, _ in sample_dict[k]:
                if target.sum() == 0:
                    continue
                preds.append(pred)
                targets.append(target)
            pred_results_list.append(torch.stack(preds, dim=-1))
            gt_segs_list.append(torch.stack(targets, dim=-1))

        # pdb.set_trace()
        val_metrics['dice'], val_metrics_std['dice_std'] = mean_dice(pred_results_list, gt_segs_list,
                                                                     self.opt['num_classes'],
                                                                     self.opt['organ_list'])
        print(val_metrics)
        print(val_metrics)
        print(val_metrics_std)
        val_assd['assd'], val_assd_std['assd_std'] = mean_assd(pred_results_list, gt_segs_list,
                                                               self.opt['num_classes'],
                                                               self.opt['organ_list'])
        print(val_assd)
        print(val_assd_std)
