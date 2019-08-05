import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Unet_2levels, Dunet_2levels, DVAE_refiner, DVAE
from utils import load_DRIVE, get_generator, log_gaussian, get_mix_coef, get_my_metrics, topo_metric
from PIL import Image

########################################################################################

# change accordingly whether you want to use GPU or CPU
USE_GPU = True
GPU_NR = 0
DEVICE = 'cuda:'+str(GPU_NR) if USE_GPU else 'cpu'

# set the pipeline to be executed and hyperparameters
MODEL = 'Unet' # can be one of Unet, Dunet, DVAE_refiner
LOSS_TYPE = 'FL' # can be one of BCE, BCEw, FL

ZDIM = 100 # number of feature maps of the 3D encoding space, matters when using the DVAE refiner model

# to use pretrained models instead of training, set to True
USE_PRETRAINED = True

# how to combine the loss terms L1 and L2, matters in models with refinement
MIX_COEFF_INIT  = 0.99
MIX_COEFF_DECAY = 0.005
MIX_COEFF_MIN   = 0.3
START_DECAY = 0

# set train parameters
EPOCHS = 150
STEPS_PER_EPOCH = 300 #number of mini-batches fed at each epoch
BATCH_SIZE = 16       #number of patches per mini-batch
PATCH_SIZE = 64       #side dimension of each patch
SEED = 7

#############################################################################################

# picking a model according to the selection
if MODEL == 'Unet':
    model = Unet_2levels().to(DEVICE)
elif MODEL == 'Dunet':
    model = Dunet_2levels().to(DEVICE)
else:
    model = DVAE_refiner(ZDIM).to(DEVICE)

# load DRIVE dataset
train_X, train_Y, train_F, train_S, test_X, test_Y, test_F = load_DRIVE(PATCH_SIZE)

# loading weights
if USE_PRETRAINED:
    path = '../weights/' + MODEL.lower() + '_' + LOSS_TYPE.lower() + '_weights.pth'
    model.load_state_dict(torch.load(path))

# training procedure
else:
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad==True], 1e-3)

    # create a train generator
    train_gen = get_generator(train_X, train_Y, train_S, PATCH_SIZE, BATCH_SIZE, SEED)

    # training routine
    model.train()
    bce_loss = nn.BCELoss(reduction='none')

    for e in range(EPOCHS):
    
        train_loss = 0
        t1_loss = 0
        t2_loss = 0

        alpha = get_mix_coef(MIX_COEFF_INIT, MIX_COEFF_DECAY, START_DECAY, MIX_COEFF_MIN, e)
        print(alpha)
    
        for s in range(STEPS_PER_EPOCH):

            optimizer.zero_grad()
            x_train, y_train = next(train_gen)

            x_train = torch.from_numpy(x_train).float().to(DEVICE)
            y_train = torch.from_numpy(y_train).float().to(DEVICE)

            # forward pass and loss calculation for the Unet model
            if MODEL == 'Unet':
                seg = model(x_train)
                loss1 = bce_loss(seg, y_train)

                if LOSS_TYPE == 'BCE':
                    weight = torch.tensor([0.5, 0.5]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)
                    loss1 *= weight_                    

                elif LOSS_TYPE == 'BCEw':
                    weight = torch.tensor([0.3, 0.7]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)
                    loss1 *= weight_

                else:
                    pvess = seg.data.view(-1)
                    pback = 1 - pvess
                    mult = torch.zeros_like(pvess)
                    mult[y_train.data.view(-1).long()==1] = pback[y_train.data.view(-1).long()==1]
                    mult[y_train.data.view(-1).long()==0] = pvess[y_train.data.view(-1).long()==0]
                    mult_ = mult.clamp(1e-5, 1-1e-5)**2
                    mult_ = mult_.view_as(y_train)
                    loss1 *= mult_

                loss1 = loss1.view(y_train.shape[0],-1).sum(dim=1)
                loss = loss1.mean()

            # forward pass and loss calculation for the Double Unet model
            elif MODEL == 'Dunet':
                seg, ref = model(x_train)
                loss1 = bce_loss(seg, y_train)
                loss2 = bce_loss(ref, y_train)

                if LOSS_TYPE == 'BCE':
                    weight = torch.tensor([0.5, 0.5]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)  
                    loss1 = loss1 * weight_
                    loss2 = loss2 * weight_

                elif LOSS_TYPE == 'BCEw':
                    weight = torch.tensor([0.3, 0.7]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)
                    loss1 = loss1 * weight_
                    loss2 = loss2 * weight_

                else:
                    weight = torch.tensor([0.5, 0.5]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)  
                    loss1 = loss1 * weight_

                    pvess = ref.data.view(-1)
                    pback = 1 - pvess
                    mult = torch.zeros_like(pvess)
                    mult[y_train.data.view(-1).long()==1] = pback[y_train.data.view(-1).long()==1]
                    mult[y_train.data.view(-1).long()==0] = pvess[y_train.data.view(-1).long()==0]
                    mult_ = mult.clamp(1e-5, 1-1e-5)**2
                    mult_ = mult_.view_as(y_train)
                    loss2 = loss2 * mult_

                loss1 = loss1.view(y_train.shape[0],-1).sum(dim=1)
                loss2 = loss2.view(y_train.shape[0],-1).sum(dim=1)
                loss = alpha * loss1.mean() + (1-alpha) * loss2.mean()

            # forward pass and loss calculation for the DVAE refiner model
            else:
                seg, mu, logvar, z, ref = model(x_train, phase='training')
                loss1 = bce_loss(seg, y_train)

                log_q_z_x = log_gaussian(z, mu, logvar)
                log_p_z   = log_gaussian(z, z.new_zeros(z.shape), z.new_zeros(z.shape))
                log_p_x_z = -bce_loss(ref, y_train)

                if LOSS_TYPE == 'BCE':
                    weight = torch.tensor([0.5, 0.5]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)
                    loss1 = loss1 *  weight_
                    log_p_x_z = log_p_x_z * weight_

                elif LOSS_TYPE == 'BCEw':
                    weight = torch.tensor([0.3, 0.7]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)
                    loss1 = loss1 * weight_ 
                    log_p_x_z = log_p_x_z * weight_

                else:
                    weight = torch.tensor([0.5, 0.5]).to(DEVICE)
                    weight_ = weight[y_train.data.view(-1).long()].view_as(y_train)
                    loss1 *= weight_

                    pvess = ref.data.view(-1)
                    pback = 1 - ref.data.view(-1)
                    mult = torch.zeros_like(pvess)
                    mult[y_train.data.view(-1).long()==1] = pback[y_train.data.view(-1).long()==1]
                    mult[y_train.data.view(-1).long()==0] = pvess[y_train.data.view(-1).long()==0]
                    mult_ = mult.clamp(1e-5, 1-1e-5)**2
                    mult_ = mult_.view_as(y_train)
                    
                    log_p_x_z *= mult_

                loss1 = loss1.view(y_train.shape[0],-1).sum(dim=1)
                log_p_x_z = log_p_x_z.view(y_train.shape[0],-1).sum(dim=1)

                loss2 = 1e-3*(log_q_z_x - log_p_z) - log_p_x_z
                loss = alpha * loss1.mean() + (1-alpha) * loss2.mean()


            train_loss += loss.item()
            t1_loss += loss1.mean().item()

            if MODEL!='Unet':
                t2_loss += loss2.mean().item()
        
            loss.backward()
            optimizer.step()

        print('Epoch: {}\nTrain loss: {:.4f} (L1: {:.4f}, L2 {:.4f})'.format(e, train_loss/(STEPS_PER_EPOCH), t1_loss/(STEPS_PER_EPOCH), t2_loss/(STEPS_PER_EPOCH)))

###########################################################################################################

# get metrics for train and test sets

print('Getting the evaluation metrics...')
model.eval()
train_segs = np.zeros(train_Y.shape)
test_segs = np.zeros(test_Y.shape)
train_infeasible = 0
train_wrong = 0
train_correct = 0
test_infeasible = 0
test_wrong = 0
test_correct = 0


with torch.no_grad():

    for i in range(train_X.shape[0]):

        img = torch.from_numpy(train_X[i,:,:]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(DEVICE)
        gt = train_Y[i,:,:]

        if MODEL=='Unet':
            seg = model(img)
        elif MODEL=='Dunet':
            _, seg = model(img)
        else:
            _,_,_,_, seg = model(img, phase='test')

        a,b,c = topo_metric(gt, seg.cpu().numpy()[0,0], 0.5, 1000)

        train_infeasible += a
        train_wrong += b
        train_correct += c   

        train_segs[i] = seg

    for i in range(test_X.shape[0]):

        img = torch.from_numpy(test_X[i,:,:]).unsqueeze(dim=0).unsqueeze(dim=0).float().to(DEVICE)
        gt = test_Y[i,:,:]

        if MODEL=='Unet':
            seg = model(img)
        elif MODEL=='Dunet':
            _, seg = model(img)
        else:
            _,_,_,_, seg = model(img, phase='test')

        a,b,c = topo_metric(gt, seg.cpu().numpy()[0,0], 0.5, 1000)

        test_infeasible += a
        test_wrong += b
        test_correct += c   

        test_segs[i] = seg

    train_auc, train_acc, train_sen, train_spe = get_my_metrics(train_segs,train_Y,train_F)
    test_auc, test_acc, test_sen, test_spe = get_my_metrics(test_segs,test_Y,test_F)


# Output results

print('Train results:')
print('AUC: {:.4f}'.format(train_auc), ', acc: {:.4f}'.format(train_acc), ', sen: {:.4f}'.format(train_sen), ', spe: {:.4f}'.format(train_spe))
print('Paths: {:d} infeasible'.format(train_infeasible), ', {:d} wrong'.format(train_wrong), ', {:d} correct'.format(train_correct))

print('Test results:')
print('AUC: {:.4f}'.format(test_auc), ', acc: {:.4f}'.format(test_acc), ', sen: {:.4f}'.format(test_sen), ', spe: {:.4f}'.format(test_spe))
print('Paths: {:d} infeasible'.format(test_infeasible), ', {:d} wrong'.format(test_wrong), ', {:d} correct'.format(test_correct))
