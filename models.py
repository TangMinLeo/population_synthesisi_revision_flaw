import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import Net_model
import logging
from torch.utils.tensorboard import SummaryWriter
import math
import json

code_map = {}

code_map['Age:[5,10)'] = '[5,10)'
code_map['Age:[10,15)'] = '[10,15)'
code_map['Age:[15,20)'] = '[15,20)'
code_map['Age:[20,25)'] = '[20,25)'
code_map['Age:[25,30)'] = '[25,30)'
code_map['Age:[30,35)'] = '[30,35)'
code_map['Age:[35,40)'] = '[35,40)'
code_map['Age:[40,45)'] = '[40,45)'
code_map['Age:[45,50)'] = '[45,50)'
code_map['Age:[50,55)'] = '[50,55)'
code_map['Age:[55,60)'] = '[55,60)'
code_map['Age:[60,65)'] = '[60,65)'
code_map['Age:[65,70)'] = '[65,70)'
code_map['Age:[70,75)'] = '[70,75)'
code_map['Age:[75,80)'] = '[75,80)'
code_map['Age:[80,85)'] = '[80,85)'
code_map['Age:[85,90]'] = '[85,90]'

code_map['Gender:1'] = 'gender:male'
code_map['Gender:2'] = 'gender:female'

code_map['Homeincome:1'] = 'Homeincome:<1 million'
code_map['Homeincome:2'] = 'Homeincome:1 million-3 million'
code_map['Homeincome:3'] = 'Homeincome:3 million-5 million'
code_map['Homeincome:4'] = 'Homeincome:5 million-10 million'
code_map['Homeincome:5'] = 'Homeincome:>10 million'

code_map['Hometype:1'] = 'Hometype:apartment'
code_map['Hometype:2'] = 'Hometype:villa'
code_map['Hometype:3'] = 'Hometype:multi-family'
code_map['Hometype:4'] = 'Hometype:single house'
code_map['Hometype:5'] = 'Hometype:dual purpose house'
code_map['Hometype:6'] = 'Hometype:other'

code_map['CarOwn:1'] = 'CarOwn:yes'
code_map['CarOwn:2'] = 'CarOwn:no'

code_map['Driver:1'] = 'Driver:yes'
code_map['Driver:2'] = 'Driver:no'

code_map['Workdays:1'] = 'Workdays:5 days per week'
code_map['Workdays:2'] = 'Workdays:6 days per week'
code_map['Workdays:3'] = 'Workdays:1-4 days per week'
code_map['Workdays:4'] = 'Workdays:inoccupation/non-regular'

code_map['Worktype:1'] = 'Worktype:student'
code_map['Worktype:2'] = 'Worktype:inoccupation/housewife'
code_map['Worktype:3'] = 'Worktype:experts'
code_map['Worktype:4'] = 'Worktype:service'
code_map['Worktype:5'] = 'Worktype:sales'
code_map['Worktype:6'] = 'Worktype:manager/office'
code_map['Worktype:7'] = 'Worktype:agriculture and fisher'
code_map['Worktype:8'] = 'Worktype:simple labor'
code_map['Worktype:9'] = 'Worktype:others'

code_map['Student:1'] = 'Student:elementary/middle/high'
code_map['Student:2'] = 'Student:kid'
code_map['Student:3'] = 'Student:university'
code_map['Student:4'] = 'Student:none'

code_map['NumHH:1'] = 'NumHH:1'
code_map['NumHH:2'] = 'NumHH:2'
code_map['NumHH:3'] = 'NumHH:3'
code_map['NumHH:4'] = 'NumHH:4'
code_map['NumHH:5'] = 'NumHH:5'
code_map['NumHH:6'] = 'NumHH:6'
code_map['NumHH:7'] = 'NumHH:7'

code_map['KidinHH:1'] = 'KidinHH:yes'
code_map['KidinHH:2'] = 'KidinHH:no'

code_map['ComMode:Walking'] = 'ComMode:Walking'
code_map['ComMode:Car'] = 'ComMode:Car'
code_map['ComMode:PM'] = 'ComMode:bike'
code_map['ComMode:None'] = 'ComMode:None'
code_map['ComMode:Taxi'] = 'ComMode:Taxi'
code_map['ComMode:PT'] = 'ComMode:public transportation'

code_map['ComTime:Peak'] = 'ComTime:Peak'
code_map['ComTime:NonPeak'] = 'ComTime:NonPeak'
code_map['ComTime:Others'] = 'ComTime:Others'
code_map['ComTime:None'] = 'ComTime:None'


class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=13, voc_size=70, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.voc_size = voc_size
        self.channels = channels
        self.device = device
        # self.emb_model = Embedding(vocab, emb_size)

        # self.beta = self.prepare_noise_schedule().to(device)
        # self.beta = self.cosine_beta_schedule().to(device)
        self.beta = self.prepare_noise_schedule_0().to(device)
        
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        print('alpha_hat[0]: ', self.alpha_hat[0])
        print('alpha_hat[-1]: ', self.alpha_hat[-1])
        print('alpha[-1]: ', self.alpha[-1])
        print('alpha[0]: ', self.alpha[0])
        print('beta[-1]: ', self.beta[-1])
        print('beta[0]: ', self.beta[0])
        print(self.beta.shape)
        

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    
    # def cosine_beta_schedule(self, s=0.008):
    #     steps = self.noise_steps + 1
    #     x = torch.linspace(0, self.noise_steps, steps)
    #     alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    #     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    #     return betas
    

    def prepare_noise_schedule_0(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alphas = 1 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        print(alphas_bar_sqrt_T)
        print(alphas_bar_sqrt_0)

        alphas_bar_sqrt = alphas_bar_sqrt-alphas_bar_sqrt_T

        alphas_bar_sqrt = alphas_bar_sqrt * alphas_bar_sqrt_0/(alphas_bar_sqrt_0-alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:]/alphas_bar[:-1]

        alphas = torch.cat([alphas_bar[0:1], alphas])

        betas = 1 - alphas
        return betas


    # 这里t是一个批量时间，针对一个batch样本，每个样本对应一个噪声时刻t
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        # Ɛ = torch.randn_like(x)
        # return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
        yup = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * yup, yup

    def sample_timesteps(self, n):
        return torch.randint(low=0, high=self.noise_steps, size=(n,))
        # return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new populations....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channels, self.voc_size)).to(self.device)
            # for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # predicted_noise = model(x, t)
                predicted_v = model(x, t)
                
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                pre_x_0 = torch.sqrt(alpha_hat)*x - torch.sqrt(1-alpha_hat)*predicted_v
                
                # if i > 1:
                #     noise = torch.randn_like(x)
                # else:
                #     noise = torch.zeros_like(x)
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                if i > 0:
                    alpha_hat_minus = self.alpha_hat[t-1][:, None, None]
                    # sigma = (1-alpha_hat_minus)/(1-alpha_hat)*beta
                    noise = torch.randn_like(x)

                    pre_u_0 = (torch.sqrt(alpha_hat_minus)*beta/(1-alpha_hat))*pre_x_0 + (torch.sqrt(alpha)*(1-alpha_hat_minus)/(1-alpha_hat))*x
                    x = pre_u_0 + torch.sqrt(beta) * noise

                    # x = pre_u_0 + torch.sqrt(beta) * noise
                else:
                    x = pre_x_0
                
        model.train()

        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # softmax = nn.Softmax(dim=-1)
        out = torch.argmax(x, dim=-1)

        return out

    def sample_acc(self, model, n):
        logging.info(f"Accelerated sampling {n} new populations....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.channels, self.voc_size)).to(self.device)
            jump_steps = list(reversed(range(0, self.noise_steps+1, 100)))
            jump_steps[0] = 999
            jump_steps[-1] = 1
            # for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
            for i in tqdm(range(len(jump_steps)), position=0):
                # print(jump_steps[i])
                t = (torch.ones(n) * jump_steps[i]).long().to(self.device)
                if jump_steps[i] > 1:
                    s = (torch.ones(n) * jump_steps[i+1]).long().to(self.device)

                    predicted_noise = model(x, t)

                    alpha = self.alpha[t][:, None, None]
                    alpha_hat_t = self.alpha_hat[t][:, None, None]
                    beta = self.beta[t][:, None, None]

                    alpha_hat_s = self.alpha_hat[s][:, None, None]

                    # noise = torch.randn_like(x)

                    x = (torch.sqrt(alpha_hat_s)/torch.sqrt(alpha_hat_t))*(x - torch.sqrt(1 - alpha_hat_t)*predicted_noise) + torch.sqrt(1 - alpha_hat_s)*predicted_noise

                    # x = (torch.sqrt(alpha_hat_s)/torch.sqrt(alpha_hat_t))*(x - torch.sqrt(1 - alpha_hat_t)*predicted_noise) + torch.sqrt(1 - alpha_hat_s)*predicted_noise
                    # + 0.1*torch.sqrt((1-alpha_hat_s)*(alpha_hat_s-alpha_hat_t)/((1-alpha_hat_t)*alpha_hat_s))*noise
                else:
                    predicted_noise = model(x, t)

                    alpha_hat_ = self.alpha_hat[t][:, None, None]
                    x = (x - torch.sqrt(1-alpha_hat_)*predicted_noise)/torch.sqrt(alpha_hat_)

                # if jump_steps[i] > 1:
                #     noise = torch.randn_like(x)
                # else:
                #     noise = torch.zeros_like(x)
                # x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()

        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        # softmax = nn.Softmax(dim=-1)
        out = torch.argmax(x, dim=-1)

        return out

def train(args):
    setup_logging(args.run_name)
    device = args.device

    file_stream = open(f'./models/{args.run_name}/voca_map.json', 'w')

    dataloader, vocabulary = get_data(args)

    new_map = {}

    for key in vocabulary.keys():
        new_map[vocabulary[key]] = key

    print(new_map)
    json.dump(new_map, file_stream, ensure_ascii=False)
    file_stream.close()

    model = Net_model(c_in=13, c_out=13, time_dim=512, voca_size=len(vocabulary), device="cuda:1").to(device)

    # ckpt = torch.load("./models/DDPM_category2/ckpt_353.pt")
    # model.load_state_dict(ckpt)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epo: 1 - epo/500)

    mse = nn.MSELoss()

    diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=13, voc_size=70, device=device)

    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    print('dataloader len', l)

    gene_dict = {}
    ff = open(f"./models/{args.run_name}/all_genes.json", "w")

    # for epoch in range(args.epochs):
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        print(f'training epoch {epoch}')
        pbar = tqdm(dataloader)
        loss_list = []
        
        for i, samples in enumerate(pbar):
            samples = samples.to(device)
            # print(samples.shape)
            # print(samples.is_cuda)
            t = diffusion.sample_timesteps(samples.shape[0]).to(device)
            # print(f't generated by sample from interval is on cuda: {t.is_cuda}')

            x_t, noise = diffusion.noise_images(samples, t)

            # print(f't after use in adding noise is on cuda: {t.is_cuda}')
            # predicted_noise = model(x_t, t)
            predicted_v = model(x_t, t)
            v_t = torch.sqrt(diffusion.alpha_hat[t][:, None, None])*noise - torch.sqrt(1-diffusion.alpha_hat[t][:, None, None])*samples
            
            loss = mse(v_t, predicted_v)

            # loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
            loss_list.append(loss.item())

        sampled_items = diffusion.sample(model, n=6)
        print(sampled_items)
        sample_out = sampled_items.detach().cpu().numpy()
        sample_out = sample_out.tolist()
        gene_rep = []
        for j in range(len(sample_out)):
            tem = []
            for item in sample_out[j]:
                tem.append(code_map[new_map[item]])
            gene_rep.append(tem)
            print(tem)
        gene_dict[epoch] = [gene_rep, np.mean(np.array(loss_list)), loss_list]

        # save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt_{epoch}.pt"))

        scheduler.step()
        print(f'learning rate: {scheduler.get_last_lr()}')

    json.dump(gene_dict, ff, ensure_ascii=False)
    ff.close()


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_category7"
    if os.path.exists(f'./models/{args.run_name}'):
        pass
    else:
        os.mkdir(f'./models/{args.run_name}')

    args.voc_size = 70
    args.emb_dim = 512

    args.epochs = 700
    args.batch_size = 32
    # args.image_size = [72, 96]
    # args.dataset_path = r"D:\Download\luce_clean_1011\train"
    args.file_path = './data_NHTSint_indiv.csv'
    args.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    args.lr = 3e-4
    # args.save_file = r"D:\我的桌面\社区调查\gene2.json"
    args.save_file = f'./models/{args.run_name}/gene2.json'
    train(args)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # diffusion = Diffusion()
    diffusion = Diffusion(noise_steps=1000, beta_start=1e-4, beta_end=0.02, channels=13, voc_size=70, device='cuda')

    model = Net_model(c_in=13, c_out=13, time_dim=512, voca_size=70, device="cuda").to(device)

    ckpt = torch.load("./models/DDPM_category5/ckpt_671.pt")
    model.load_state_dict(ckpt)

    out = diffusion.sample_acc(model, 300)
    # out = diffusion.sample(model, 300)

    print(out)
    f = open('./models/DDPM_category5/voca_map.json', 'r')

    data = json.load(f)
    f.close()
    # print(data)
    sample_out = out.detach().cpu().numpy()
    sample_out = sample_out.tolist()
    for i in range(len(sample_out)):
        tem = []
        for j in range(len(sample_out[i])):
            tem.append(data[str(sample_out[i][j])])
        print(tem)
