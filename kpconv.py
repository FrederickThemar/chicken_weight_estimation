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
        self.GPU_ID = '0' # GPU to be used
        
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
        self.loop_times = []

        print("Done!\n")

    def estimate_frame(self, pcds):
        # Log start time of function
        start = time.time()

        outputs = []

        # Create Dataloader
        test_dataset = ChickenWeightDataset(self.config, pcds, train=False)
        test_sampler = ChickenWeightSampler(test_dataset)
        collate_fn = ChickenWeightCollate

        test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                sampler=test_sampler,
                                collate_fn=collate_fn,
                                num_workers=self.config.input_threads,
                                pin_memory=True)

        # Step 2: Feeding pcd to network
        count = 0
        loop_start = time.time()
        for batch in test_loader:
            # DEBUG: Record time taken for the loop
            loop_time = time.time() - loop_start
            self.loop_times.append(loop_time)

            count+=1
            batch.to(self.device)
            output = self.net(batch, self.config)

        output = output.cpu().detach().numpy()
        outputs.append(output)

        # Save process time
        end = time.time()
        duration = (end - start)
        self.times.append(duration)

        return outputs

    def estimate_video(self, pcdVideoPath):
        pcds = [self.read_pcd(pcdPath) for pcdPath in pcdVideoPath]
        pcds = torch.stack(pcds, dim=0)  # TxNx3
        
        batch_size = 16
        for i in range(0, pcds.shape[0], batch_size):
            batch = pcds[i:i+16]
            batch = batch.cuda()

