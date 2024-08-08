import os
import time
import torch
import numpy as np

# Dataset
from KPConv.datasets.ChickenWeight import *
from KPConv.datasets.ModelNet40 import *
from KPConv.datasets.S3DIS import *
from KPConv.datasets.SemanticKitti import *
from torch.utils.data import DataLoader

from KPConv.utils.config import Config
from KPConv.utils.tester import ModelTester
from KPConv.archs.architectures import KPCNN, KPFCNN, KPCNN_LinReg

class kpconv():
    def __init__(self):
        print("Initializing weight estimation model... ", end="", flush=True)

        self.chosen_log = './KPConv/models/Log_2024-07-20_22-13-05'
        self.GPU_ID = '1' # GPU to be used
        
        # Set GPU visible device
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU_ID

        # Find all checkpoints in the chosen training folder
        self.chkp_path = os.path.join(self.chosen_log, 'checkpoints', 'current_chkp.tar')

        # Initialize configuration class
        self.config = Config()
        self.config.load(self.chosen_log)

        self.config.validation_size = 2000
        self.config.input_threads = 10

        self.orient_correction = True

        self.net = KPCNN_LinReg(self.config)

        # Choose to train on CPU or GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.net.to(self.device)
        
        checkpoint = torch.load(self.chkp_path,map_location='cuda:0', weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()

        self.times = []
        self.dataset_times = []
        self.sampler_times = []
        self.collate_times = []
        self.loader_times =  []
        self.loop_times =    []
        self.output_times =  []

        print("Done!\n")

    # def read_pcd(self, pcdPath):
    #     pcd = np.asarray(o3d.io.read_point_cloud(pcdPath).points).astype(np.float32)
    #     # Subsample them
    #     if self.config.first_subsampling_dl > 0:
    #         point = grid_subsampling(pcd[:, :3], sampleDl=self.config.first_subsampling_dl)
    #     else:
    #         point = pcd[:, :3]
    #     points = []
    #     points+=[point]
    #     if self.orient_correction:
    #         points = [pp[:, [0, 2, 1]] for pp in points]

    #     # points = torch.tensor(points)
    #     return points

    def estimate_frame(self, pcdPath):
        # Log start time of function
        start = time.time()

        # Create Dataloader
        test_dataset = ChickenWeightDataset(self.config, pcdPath, train=False)

        dataset_end = time.time()
        dataset_time = dataset_end - start
        self.dataset_times.append(dataset_time)
        sampler_start = time.time()

        test_sampler = ChickenWeightSampler(test_dataset)

        sampler_end = time.time()
        sampler_time = sampler_end - sampler_start
        self.sampler_times.append(sampler_time)
        collate_start = time.time()

        collate_fn = ChickenWeightCollate

        collate_end = time.time()
        collate_time = collate_end - collate_start
        self.collate_times.append(collate_time)
        loader_start = time.time()

        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 sampler=test_sampler,
                                 collate_fn=collate_fn,
                                 num_workers=self.config.input_threads,
                                 pin_memory=True)

        loader_end = time.time()
        loader_time = loader_end - loader_start
        self.loader_times.append(loader_time)

        # Step 2: Feeding pcd to network
        count = 0
        loop_start = time.time()
        for batch in test_loader:
            count+=1
            batch.to(self.device)
            output = self.net(batch, self.config)
        loop_end = time.time()
        loop_time = loop_end - loop_start
        self.loop_times.append(loop_time)
        output_start = time.time()

        output = output.cpu().detach().numpy()

        output_end = time.time()
        output_time = output_end - output_start
        self.output_times.append(output_time)

        # Save process time
        end = time.time()
        duration = (end - start)
        self.times.append(duration)

        return output

    def estimate_video(self, pcdVideoPath):
        pcds = [self.read_pcd(pcdPath) for pcdPath in pcdVideoPath]
        pcds = torch.stack(pcds, dim=0)  # TxNx3
        
        batch_size = 16
        for i in range(0, pcds.shape[0], batch_size):
            batch = pcds[i:i+16]
            batch = batch.cuda()

