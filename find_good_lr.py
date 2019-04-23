import math
import matplotlib.pyplot as plt
'''
more detail can be found here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
example:
run:
    net=YourNet()
    train_dataset=YourDataset
    train_loader = Data.DataLoader(dataset=train_dataset,shuffle=opt.shuffle_dataset,
                                       batch_size=20, num_workers=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)

then watch the saved picture to find a good initial learning rate                    
'''

def find_good_lr(trn_loader,optimizer,net,criterion,
                 init_value = 1e-8, final_value=1., beta = 0.98):
    #form:
    #https://github.com/sgugger/Deep-Learning
    #https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    num = round(len(trn_loader)-1)
    mult = 1.05
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    iter=0
    for data in trn_loader:
        if iter == num or lr >= final_value:
            break
        batch_num += 1
        # As before, get the loss for this mini-batch of inputs/outputs
        inputs = data['image'].cuda()
        labels = data['big_cate'].cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.data
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        loss.backward()
        print('iter: %d  loss: %.5f   lr: %.8f'%(iter,loss.item(),lr))
        optimizer.step()
        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        iter+=1
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.savefig("./lr_finding.png")
        try:
            plt.show()
        except:
            print('can not show plt figure,saving lr picture...')
    return log_lrs, losses
