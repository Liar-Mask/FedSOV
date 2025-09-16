import os
from utils.args import parser_args
# from utils.help import *
from utils.datasets import *
import copy
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
#from dataset import prepare_dataset, prepare_wm
#from utils.help import *
import time
import datetime
import torch.optim as optim


from experiments.base import Experiment
from experiments.trainer_private import TrainerPrivate, TesterPrivate
from experiments.utils import construct_passport_kwargs
from experiments.byz_defences import *
from models.alexnet import AlexNet
from models.layers.conv2d import ConvBlock
from models.resnet import ResNet18


class IPRFederatedLearning(Experiment):
    """
    Perform federated learning
    """
    def __init__(self, args):
        super().__init__(args) # define many self attributes from args
        self.criterion = torch.nn.CrossEntropyLoss()
        self.in_channels = 3

        if args.dataset == 'cifar10' or args.dataset == 'minist':
            self.num_labels=10
        if args.dataset == 'cifar100':
            self.num_labels=100
        
        self.num_bit = args.num_bit
        self.num_trigger = args.num_trigger
        self.dp = args.dp
        self.sigma = args.sigma

        data_root = '/CIS32/zgx/Fed2/Data/'
 
        print('==> Preparing data...')
        self.train_set, self.test_set, self.dict_users = get_data(dataset=self.dataset,
                                                        data_root = self.data_root,
                                                        iid = self.iid,
                                                        num_users = self.num_users,
                                                        beta=args.beta
                                                        )
     
        #print('==> Preparing watermark..')
        if args.backdoor_indis:
            if args.dataset == 'cifar10':

                self.wm_data, self.wm_dict = prepare_wm_new(data_root + 'trigger/cifar10/', self.num_back, self.num_trigger)
            if args.dataset == 'cifar100':
                self.wm_data, self.wm_dict = prepare_wm_indistribution(data_root + 'trigger/cifar100/', self.num_back, self.num_trigger)
        else:
            self.wm_data, self.wm_dict = prepare_wm(data_root + 'trigger/pics', self.num_back)
        

        if self.weight_type == 'gamma':
            if self.loss_type == 'sign':
                self.scheme = 0
            if self.loss_type == 'CE':
                self.scheme = 1
        
        if self.weight_type == 'kernel':
            if self.loss_type == 'sign':
                self.scheme = 2
            if self.loss_type == 'CE':
                self.scheme = 3 

        print('==> Preparing model...')

        self.logs = {'train_acc': [], 'train_sign_acc':[], 'train_wm_acc': [], 'train_loss': [], 'train_sign_loss': [],
                     'val_acc': [], 'val_loss': [],
                     'test_acc': [], 'test_loss': [],
                     'keys':[],
                     'trigger_dict': self.wm_dict,

                     'best_test_acc': -np.inf,
                     'best_model': [],
                     'local_loss': [],
                     }
        self.embed_mat_dict=construct_passport_kwargs(self) #各客户端统一水印和提取矩阵 #嵌入矩阵字典self.embed_mat_dict=construct_passport_kwargs(self) #各客户端统一水印和提取矩阵 #嵌入矩阵字典

        self.generate_signature_dict()
        self.construct_model()
        
        self.w_t = copy.deepcopy(self.model.state_dict())

        self.trainer = TrainerPrivate(self.model, self.device, self.dp, self.sigma)
        self.tester = TesterPrivate(self.model, self.device)

        self.makedirs_or_load()
    
    def generate_signature_dict(self):
        
        l = []
        for i in range(self.num_users):
            if i < self.num_sign:
                l.append(1)
            else:
                l.append(0)
        
        np.random.shuffle(l)
        self.keys = []       
        for i in range(self.num_users):
            if l[i] == 1:
                #key = construct_passport_kwargs(self)
                self.keys.append(self.embed_mat_dict)
            if l[i] == 0:
                self.keys.append(None)

        self.logs['keys'].append(self.keys)
              
    def construct_model(self):

        self.passport_kwargs = self.embed_mat_dict

        if self.model_name == 'alexnet':
            model = AlexNet(self.in_channels, self.num_classes, self.passport_kwargs)
        else:
            model = ResNet18( passport_kwargs = self.passport_kwargs)
        
        self.model = model.to(self.device)

    def train(self):
        # these dataloader would only be used in calculating accuracy and loss
        train_ldr = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False, num_workers=2)
        val_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)
        test_ldr = DataLoader(self.test_set, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)
        wm_test_ldr = DataLoader(self.wm_data, batch_size=self.batch_size * 2, shuffle=False, num_workers=2)

        local_train_ldrs = []
        wm_loaders = []
        if args.iid:
            
            for i in range(self.num_users):
                local_train_ldr = DataLoader(DatasetSplit(self.train_set, self.dict_users[i]), batch_size = self.batch_size,
                                                shuffle=True, num_workers=2)
                local_train_ldrs.append(local_train_ldr)
        else:
            for i in range(self.num_users):
                local_train_ldr = DataLoader(self.dict_users[i], batch_size = self.batch_size,
                                                shuffle=True, num_workers=2)
                local_train_ldrs.append(local_train_ldr)

        for i in range(self.num_back):
            #print(self.wm_dict[i])
            #print(len(self.wm_dict[i]))
            wm_loader = DataLoader(DatasetSplit(self.wm_data, self.wm_dict[i]), batch_size=2, shuffle=True, num_workers =4, drop_last=True)
            wm_loaders.append(wm_loader)

        total_time=0
        file_name = "_".join(
                ['FedSOV_log', f's{args.seed}',str(args.num_bit), str(args.batch_size),str(args.lr), str(args.iid), str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))])
        log_dir=os.getcwd() + '/'+args.log_dir+'/'+ args.model_name +'/'+ args.dataset

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_fn=log_dir+'/'+file_name+'.log'

        print(log_fn)
        for epoch in range(self.epochs):
            if self.sampling_type == 'uniform':
                # print("frac:",self.frac )
                self.m = max(int(self.frac * self.num_users), 1)
                idxs_users = np.random.choice(range(self.num_users), self.m, replace=False)

            local_ws, local_losses, sign_losses, private_sign_acces, acc_wms = [], [], [], [], []
            grad_cat_lists=[]
            start = time.time()
            test_num=0
            for idx in tqdm(idxs_users, desc='Epoch:%d, lr:%f' % (self.epochs, self.lr)):
                self.model.load_state_dict(self.w_t)
     
                if idx < self.num_back:
                    local_w, local_loss, sign_loss = self.trainer._local_update(local_train_ldrs[idx], wm_loaders[idx], self.local_ep, self.lr, self.keys[idx], self.scheme)
                else:
                    wm_loader = None
                    if self.args.byz_attack=="label_flipping" and idx > self.args.num_users-self.args.num_byz:
                        flipping_mark=True
                    else:
                        flipping_mark=False
                    local_w, grad_cat_list, local_loss, sign_loss, local_acc = self.trainer._local_update_noback(local_train_ldrs[idx], self.num_labels, flipping_mark, self.local_ep, self.lr, self.keys[idx], self.scheme)
                    if test_num==0:
                        local_train_mean, local_acc_train_mean = self.trainer.test(train_ldr)
                        local_val_mean, local_acc_val_mean = self.trainer.test(val_ldr)
                    if test_num==0:
                        test_num+=1
                        print('id:{}  ****local_train_acc:{:.4f}, ****local_val_acc:{:.4f}'.format(idx, local_acc_train_mean,local_acc_val_mean))

                local_ws.append(copy.deepcopy(local_w))
                grad_cat_lists.append(copy.deepcopy(grad_cat_list))
                local_losses.append(local_loss)
                sign_losses.append(sign_loss)

            #client_weights = np.ones(self.m) / self.m
            client_weights = []
            for i in range(self.num_users):
                client_weight = len(DatasetSplit(self.train_set, self.dict_users[i]))/len(self.train_set)
                client_weights.append(client_weight)
            
            if self.args.byz_attack !=None:
                new_grad_list=[]
                for mac in range(self.args.num_users):
                    if self.args.num_byz > 0:
                        # Byzantine Machines
                        if mac >= self.args.num_users - self.args.num_byz:
                            byz_grad = grad_cat_lists[mac]
                            if self.args.byz_attack == 'gaussion':
                                byz_grad = torch.normal(10, 100, byz_grad.shape).cuda(self.device)

                            new_grad_list.append(byz_grad)
                        else:
                            new_grad_list.append(grad_cat_lists[mac])
                    else:
                        new_grad_list.append(grad_cat_lists[mac])
                grad_cat_lists=new_grad_list    

            self._fed_avg(local_ws, grad_cat_lists, self.lr, self.args.defence_type, client_weights, self.args.num_byz, epoch)
            self.lr = self.lr * 0.99

            self.model.load_state_dict(self.w_t)
            end = time.time()
            interval_time = end - start
            total_time+=interval_time
            if (epoch + 1) == self.epochs or (epoch + 1) % 1 == 0:
                loss_train_mean, acc_train_mean = self.trainer.test(train_ldr)
                loss_val_mean, acc_val_mean = self.trainer.test(val_ldr)
                loss_test_mean, acc_test_mean = loss_val_mean, acc_val_mean
                

                # test for watermarks
                if self.num_back>0:
                    for i in range(self.num_back):
                        wm_loader = DataLoader(DatasetSplit(self.wm_data, self.wm_dict[i]), batch_size=2, shuffle=True, num_workers =2, drop_last=True)
                        loss_wm, acc_wm = self.trainer.test(wm_loader)
                        acc_wms.append(acc_wm)
                else: 
                    acc_wm = 0

                for idx in range(self.num_users):
                    private_sign_acc = self.tester.test_signature(self.keys[idx], self.scheme)
                    private_sign_acces.append(private_sign_acc)
                
                #print(private_sign_acces)

                self.logs['train_acc'].append(acc_train_mean)
                self.logs['train_loss'].append(loss_train_mean)
                self.logs['train_sign_acc'].append(private_sign_acces) 
                self.logs['train_wm_acc'].append(acc_wm)

                self.logs['val_acc'].append(acc_val_mean)
                self.logs['val_loss'].append(loss_val_mean)
                self.logs['local_loss'].append(np.mean(local_losses))


                # use validation set as test set
                if self.logs['best_test_acc'] < acc_val_mean:
                    self.logs['best_test_acc'] = acc_val_mean
                    self.logs['best_test_loss'] = loss_val_mean
                    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())

                print('Epoch {}/{}  --time {:.1f}s'.format(
                    epoch, self.epochs,
                    interval_time
                )
                )
                mean_sign_loss = 0.
                mean_private_sign_acc = 0.
                mean_acc_wm = 0.

                for i in sign_losses:
                    mean_sign_loss += i
                mean_sign_loss /= len(sign_losses)

                for i in acc_wms:
                    mean_acc_wm += i
                if self.num_back != 0:
                    mean_acc_wm /= self.num_back
                
                for i in private_sign_acces:
                    if i != None:
                        mean_private_sign_acc += i
                
                if self.num_sign != 0:
                    mean_private_sign_acc /= self.num_sign

                print(
                    "Train Loss {:.4f} --- Val Loss {:.4f} --- Sign Loss {:.4f} --- Private Sign Acc {:.4f}"
                    .format(loss_train_mean, loss_val_mean, mean_sign_loss, mean_private_sign_acc))
                print("Train acc {:.4f} --- Val acc {:.4f} --Best acc {:.4f}".format(acc_train_mean, acc_val_mean,
                                                                                                        self.logs[
                                                                                                            'best_test_acc']
                                                                                                        )
                      )
                s = 'epoch:{}, lr:{:.4f}, train_acc:{:.4f}, val_acc:{:.4f}, sign Acc:{:.4f}, time:{:.4f}, total_time:{:.4f}'.format(epoch,self.lr,acc_train_mean,acc_val_mean,mean_private_sign_acc,interval_time,total_time)
                
                with open(log_fn,"a") as f:
                    json.dump({"epoch":epoch,"train_acc":round(acc_train_mean,4  ),"test_acc":round(acc_val_mean,4),"sign_acc":round(mean_private_sign_acc,4),"time":round(total_time,2)},f)
                    f.write('\n')
            
            today = datetime.date.today()
            if (epoch+1) % 100==0:
                save_dir =os.getcwd() + f'/{self.args.log_dir}/' + self.args.model_name +'/' + self.args.dataset
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                pkl_name = "_".join(
                            ['FedSOV_pkl', f's{self.args.seed}',f'e{epoch}', str(args.num_bit), str(args.batch_size),str(args.lr), str(args.iid), f'{today.year}_{today.month}_{today.day}'])
                pkl_name=save_dir+'/'+ pkl_name
                print("pkl_name:",pkl_name)
                torch.save( copy.deepcopy(self.model.state_dict()),pkl_name+".pkl")

        print('------------------------------------------------------------------------')
        print('Test loss: {:.4f} --- Test acc: {:.4f} --- Watermark acc{:.4f} --- Sign acc{:.4f}'.format(self.logs['best_test_loss'], 
                                                                                       self.logs['best_test_acc'],
                                                                                       acc_wm, mean_private_sign_acc))

        return self.logs, interval_time, self.logs['best_test_acc'], acc_test_mean, acc_wm, mean_private_sign_acc


    def _fed_avg_(self, local_ws, grad_cat_lists, lr, defence_type, client_weights, num_byz, epoch):

        if defence_type==None:
            client_weights=1.0/len(local_ws)
            w_avg = copy.deepcopy(local_ws[0])
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] * client_weights

                for i in range(1, len(local_ws)):
                    w_avg[k] += local_ws[i][k] * client_weights

        self.w_t = local_ws[0]
        

    def _fed_avg(self, local_ws, grad_cat_lists, lr, defence_type, client_weights, num_byz, epoch):

        if defence_type==None:
            print('avg...')
            client_weights=1.0/len(local_ws)
            w_avg = copy.deepcopy(local_ws[0])
            for k in w_avg.keys():
                w_avg[k] = w_avg[k] * client_weights

                for i in range(1, len(local_ws)):
                    w_avg[k] += local_ws[i][k] * client_weights

                self.w_t[k] = w_avg[k]
        elif defence_type == 'median':
            self.model.load_state_dict(self.w_t)
            self.optimizer = optim.SGD(self.model.parameters(),
                lr,
                momentum=0.9,
                weight_decay=0.0005) 
            param_list_ = []
            for each_param_list in grad_cat_lists:
                each_param_array = each_param_list.squeeze()
            param_list_.append(each_param_array)

            grad_array = torch.cat([x.reshape((-1, 1)) for x in param_list_], dim=1)
            med, _ = torch.median(grad_array, axis=1)
            #med=grad_array[5]
            
            defence_update(self.model, med)
            self.optimizer.step() 
            self.w_t=self.model.state_dict()      
        
        elif defence_type == 'trimmed':
            
            param_list_ = []
            self.model.load_state_dict(self.w_t)
            self.optimizer = optim.SGD(self.model.parameters(),
                lr,
                momentum=0.9,
                weight_decay=0.0005) 
            for each_param_list in grad_cat_lists:
                each_param_array = each_param_list.squeeze()
                param_list_.append(each_param_array)

            grad_array = torch.cat([x.reshape((-1, 1)) for x in param_list_], dim=1)
            med, _ = torch.sort(grad_array, axis=1)
            #trimmed_num = int(self.args.num_users * 0.1)
            trimmed_num = num_byz
            med = torch.mean(med[:, trimmed_num:-trimmed_num], axis=1)

            defence_update(self.model, med)
            self.optimizer.step() 
            self.w_t=self.model.state_dict()


        elif defence_type == 'fltrust':
            self.model.load_state_dict(self.w_t)
            self.optimizer = optim.SGD(self.model.parameters(),
                lr,
                momentum=0.9,
                weight_decay=0.0005) 

            global_update, grad_ag_ls=fltrust_agg(grad_cat_lists, t=epoch)
            defence_update(self.model, global_update)
            self.optimizer.step() 
            self.w_t=self.model.state_dict()

        elif defence_type == 'krum':
            param_list_ = []
            self.model.load_state_dict(self.w_t)
            self.optimizer = optim.SGD(self.model.parameters(),
                lr,
                momentum=0.9,
                weight_decay=0.0005) 

            # 计算距离i最近的n-f-2个点, 并计算score(i)
            norm_ls=torch.empty(self.args.num_users,self.args.num_users).to(self.device)
            score_kr=torch.empty(self.args.num_users).to(self.device)
            for i in range(len(grad_cat_lists)):
                for j in range(len(grad_cat_lists)):
                        norm_ls[i][j]=torch.norm(torch.sub(grad_cat_lists[i],grad_cat_lists[j])).to(self.device)
                torch.sort(norm_ls[i])
                score_kr[i]=torch.sum(norm_ls[i][0:self.args.num_users-num_byz-1]).to(self.device)

            _, krum_client=torch.sort(score_kr)

            defence_update(self.model, grad_cat_lists[krum_client[0]])
            self.optimizer.step() 
            self.w_t=self.model.state_dict()


def main(args):
    logs = {'net_info': None,
            'arguments': {
                'frac': args.frac,
                'local_ep': args.local_ep,
                'local_bs': args.batch_size,
                'lr_outer': args.lr_outer,
                'lr_inner': args.lr,
                'iid': args.iid,
                'wd': args.wd,
                'optim': args.optim,      
                'model_name': args.model_name,
                'dataset': args.dataset,
                'log_interval': args.log_interval,                
                'num_classes': args.num_classes,
                'epochs': args.epochs,
                'num_users': args.num_users
            }
            }

    fl = IPRFederatedLearning(args)

    logg, total_time, best_test_acc, test_acc, acc_wm, acc_sign = fl.train()                                         
                                             
    logs['net_info'] = logg  #logg=self.logs,    self.logs['best_model'] = copy.deepcopy(self.model.state_dict())
    logs['test_acc'] = test_acc
    logs['bp_local'] = True if args.bp_interval == 0 else False

    #print(logg['keys'])
    save_dir =os.getcwd() + f'/{args.log_dir}/' + args.model_name +'/' + args.dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    pkl_name = "_".join(
                ['FedSOV_final', str(args.num_bit), str(args.batch_size),str(args.lr), str(args.iid), str(time.strftime("%Y_%m_%d_%H%M%S", time.localtime()))])
    pkl_name=save_dir+'/'+ pkl_name
    print("pkl_name:",pkl_name)
    torch.save(logs,
            #    save_dir + args.model_name +'/' + args.dataset + '/Dp_{}_{}_iid_{}_num_sign_{}_w_type_{}_loss_{}_B_{}_alpha_{}_num_back_{}_type_{}_T_{}_epoch_{}_E_{}_u_{}_{:.1f}_{:.4f}_{:.4f}_wm_{:.4f}_sign_{:.4f}.pkl'.format(
            #        args.dp, args.sigma, args.iid, args.num_sign, args.weight_type, args.loss_type, args.num_bit, args.loss_alpha, args.num_back, args.backdoor_indis, args.num_trigger, args.epochs, args.local_ep, args.num_users, args.frac, time, test_acc, acc_wm, acc_sign
               pkl_name+".pkl"
               )
    return

if __name__ == '__main__':
    args = parser_args()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main(args)