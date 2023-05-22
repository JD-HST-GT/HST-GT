from Utils.utils_new import get_train_data,get_graph,get_y_time
import os
import torch
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from Model.HST_GT import HGT
from Utils.Metric_utils import *

def train(Recordpath):
    # split train test
    N_train = int(240 * 0.8)
    N_test = int(240 * 0.2)

    train_range = list(range(N_train))
    test_range = list(range(N_train,N_train+N_test))
    import numpy as np
    y_all_t_all,y_wh_t_all,y_pack_t_all,y_sort_t_all = get_y_time()
    y_all_t_all = torch.from_numpy(np.array(y_all_t_all)).to('cpu').to(torch.float32)
    y_wh_t_all = torch.from_numpy(np.array(y_wh_t_all)).to('cpu').to(torch.float32)
    y_pack_t_all = torch.from_numpy(np.array(y_pack_t_all)).to('cpu').to(torch.float32)
    y_sort_t_all = torch.from_numpy(np.array(y_sort_t_all)).to('cpu').to(torch.float32)

    train_data_wh,train_data_sort ,train_data_background_wh,train_data_background_sort\
    ,y_all_t,y_wh_t ,y_sort_t ,y_pack_t \
    ,wh_mask,sort_mask,wh_pack_mask,sort_pack_mask\
    ,downstream_mask,context_mask\
    ,graph = get_train_data()


    model = HGT(hidden_channels=128,  num_heads=4, num_layers=3,temporal_features=4,back_features=3,data=graph)

    device = 'cuda:0'


    


    train_data_wh = torch.from_numpy(train_data_wh).to('cuda:0').to(torch.float32)
    train_data_sort = torch.from_numpy(train_data_sort).to('cuda:0').to(torch.float32)
    train_data_background_wh = torch.from_numpy(train_data_background_wh).to('cuda:0').to(torch.float32)
    train_data_background_sort = torch.from_numpy(train_data_background_sort).to('cuda:0').to(torch.float32)



    wh_mask = wh_mask.to(device).to(torch.float32)
    sort_mask = sort_mask.to(device).to(torch.float32)
    mask_in = {
        'Node1':wh_mask,
        'Node2':sort_mask
    }
    wh_pack_mask = wh_pack_mask.to(device).to(torch.float32)
    sort_pack_mask = sort_pack_mask.to(device).to(torch.float32)
    mask_pack_in = {
        'Node1':wh_pack_mask,
        'Node2':sort_pack_mask
    }
    downmask_ins =torch.from_numpy(downstream_mask).to(device).to(torch.float32)
    cmask_ins=torch.from_numpy(context_mask).to(device).to(torch.float32)
    import numpy as np
    y_all_ts = [] 
    y_wh_ts = []
    y_sort_ts = []
    y_pack_ts = []

    for i in range(240):
        
        y_all_ts.append(torch.from_numpy(np.array(y_all_t[i])).to(device).to(torch.float32).view(-1,1))
        y_wh_ts.append(torch.from_numpy(np.array(y_wh_t[i])).to(device).to(torch.float32).view(-1,1))
        y_sort_ts.append(torch.from_numpy(np.array(y_sort_t[i])).to(device).to(torch.float32).view(-1,1))
        y_pack_ts.append(torch.from_numpy(np.array(y_pack_t[i])).to(device).to(torch.float32).view(-1,1))

    print('------------Load Data Success------------')



    import datetime
    model.train()
    model.cuda()
    model.float()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.00001)

    lossfunction = torch.nn.MSELoss()
    x_dict = graph.x_dict
    edge_index_dict = graph.edge_index_dict
    starttime = datetime.datetime.now()

    if_first = True
    loss_whs = 0
    loss_packs =0
    loss_sorts = 0
    loss_alls = 0
    losss = 0
    sn = 0
    firstcat = True
    outall = None
    outwh = None
    outpack = None
    outsort = None
    cmask = cmask_ins
    dmask = downmask_ins

    for epoch in range(0,6200):
        
        model.train()
        loss_whs = 0
        loss_packs =0
        loss_sorts = 0
        loss_alls = 0
        losss = 0
        sn = 0
        firstcat = True
        outall = None
        outwh = None
        outpack = None
        outsort = None
    
        for t in train_range:
            
            
            train_temporal_data = {
                'Node1':train_data_wh[t].view(-1,4),
                'Node2':train_data_sort[t].view(-1,4)
            }
            train_back_data = {
                'Node1':train_data_background_wh[t].view(-1,3),
                'Node2':train_data_background_sort[t].view(-1,3)
            }


            
            y_wh_t = y_wh_ts[t]
            y_pack_t = y_pack_ts[t]
            y_sort_t = y_sort_ts[t]
            y_all_t = y_all_ts[t]

        
            out = model(x_dict, edge_index_dict,train_temporal_data,train_back_data,mask_in,mask_pack_in,if_first,cmask,dmask)


            whout_norm = out['Node1']
            

            packout_norm = out['pack']
            

            sortout_norm = out['Node2']
            

            allout_norm = out['Node1'] + out['pack'] + out['Node2']
            

            if firstcat:
                firstcat = False
                outall = allout_norm.cpu().detach()
                outwh = whout_norm.cpu().detach()
                outpack = packout_norm.cpu().detach()
                outsort = sortout_norm.cpu().detach()
            else:
                outall = torch.cat((outall,allout_norm.cpu().detach()),dim=0)
                outwh = torch.cat((outwh,whout_norm.cpu().detach()),dim=0)
                outpack = torch.cat((outpack,packout_norm.cpu().detach()),dim=0)
                outsort = torch.cat((outsort,sortout_norm.cpu().detach()),dim=0)
            loss_wh = lossfunction(out['Node1'],y_wh_t)
            loss_sort = lossfunction(out['Node2'],y_sort_t)
            loss_pack = lossfunction(out['pack'],y_pack_t)
            loss_all = lossfunction(out['Node1']+out['Node2']+out['pack'],y_all_t)
            loss = loss_wh + loss_sort + loss_all + loss_pack
            
            optimizer.zero_grad()
            

            sn += len(y_all_t)
            tn = len(y_all_t)

            loss_whs += float(loss_wh) * tn
            loss_packs += float(loss_pack) * tn
            loss_sorts += float(loss_sort) * tn
            loss_alls += float(loss_all) * tn
            losss += float(loss) * tn

            loss.backward()
            optimizer.step()
            if_first = False

        pres = {
            "all":outall,
            "store":outwh,
            "pack":outpack,
            "sort":outsort
        }
        yts =  {
            "all":y_all_t_all[0:312*N_train],
            "store":y_wh_t_all[0:312*N_train],
            "pack":y_pack_t_all[0:312*N_train],
            "sort":y_sort_t_all[0:312*N_train]
        }

        
        
        if epoch % 2 == 0:
            with open(Recordpath + "train_record.txt",'a+') as f:
                print("epoch: {}, loss_wh: {}, loss_pack: {}, loss_sort: {}, loss_all: {}, loss: {}".format(epoch,loss_whs/sn,loss_packs/sn,loss_sorts/sn,loss_alls/sn,losss/sn),file=f)
            with open(Recordpath + "train_metric.txt","a+") as f:
                print(metric_all(pres,yts),file=f)
        if epoch % 10 == 0:

            print("-------train loss--------")
            endtime = datetime.datetime.now()
            print("epoch: {}, loss_wh: {}, loss_pack: {}, loss_sort: {}, loss_all: {}, loss: {}, time: {}s".format(epoch,loss_whs/sn,loss_packs/sn,loss_sorts/sn,loss_alls/sn,losss/sn,(endtime-starttime).seconds))
            starttime = datetime.datetime.now()
            r = metric_all(pres,yts)
            print("-------train metric--------")
            for i in r:
                print(i,r[i],end=' ')
            print()
               
                   
        if epoch % 50 == 0:
            torch.save(model,"Model_Save/{}_{}.pkl".format(epoch,losss/sn))
    torch.save(model,"Model_Save/model.pkl")



