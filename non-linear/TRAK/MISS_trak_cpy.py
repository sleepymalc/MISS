import sys
sys.path.append("..")

import torch
from torch.utils.data import DataLoader
from .projector import CudaProjector, ProjectionType, BasicProjector
from .grad_calculator import count_parameters, grad_calculator, out_to_loss_grad_calculator
from model_train import SubsetSamper
from tqdm import tqdm

class MISS_TRAK:
    def __init__(self,
                 model,
                 model_checkpoints,
                 train_loader,
                 test_loader,
                 model_output_class,
                 proj_dim,
                 device):
        '''
        :param model: a nn.module instance, no need to load any checkpoint
        :param model_checkpoints: a list of checkpoint path, the length of this list indicates the ensemble model number
        :param train_loader: train samples in a data loader
        :param train_loader: test samples in a data loader
        :param model_output_class: a class definition inheriting BaseModelOutputClass
        :param proj_dim: projection dimension used for reducing the dimension of the grads
        :param device: the device running
        '''
        self.model = model
        self.model_checkpoints = model_checkpoints
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_output_class = model_output_class
        self.proj_dim = proj_dim
        self.device = device

        self.TRAK_feature_cache = None

    def TRAK_feature(self):
        # Check if the result is already cached
        if self.TRAK_feature_cache is not None:
            return self.TRAK_feature_cache

        all_grads_p_list = []
        Q_list = []

        for checkpoint_id, checkpoint_file in enumerate(tqdm(self.model_checkpoints)):
            self.model.load_state_dict(torch.load(checkpoint_file))
            self.model.eval()

            print(self.model)
            print("#Parameters:", count_parameters(self.model))

            parameters = list(self.model.parameters())
            normalize_factor = torch.sqrt(torch.tensor(count_parameters(self.model), dtype=torch.float32))

            # projection of the grads
            # projector = CudaProjector(grad_dim=count_parameters(self.model), proj_dim=self.proj_dim, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)
            projector = BasicProjector(grad_dim=count_parameters(self.model), proj_dim=self.proj_dim, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)

            # Go through the training loader to get grads
            # Φ
            all_grads_p = grad_calculator(data_loader=self.train_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)
            #Q
            out_to_loss_grads = out_to_loss_grad_calculator(data_loader=self.train_loader, model=self.model, func=self.model_output_class.get_out_to_loss_grad)
            # ϕ
            all_grads_test_p = grad_calculator(data_loader=self.test_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)

            # Append to list for later averaging
            all_grads_p_list.append(all_grads_p)
            Q_list.append(out_to_loss_grads)

        # Convert lists to tensors
        all_grads_p_tensor = torch.stack(all_grads_p_list)
        Q_tensor = torch.stack(Q_list)

        # Store the result in the cache
        self.TRAK_feature_cache = (all_grads_test_p, all_grads_p_tensor, Q_tensor)

        return all_grads_test_p, all_grads_p_tensor, Q_tensor

    def most_k(self, k):
        '''
        Select the most influential k samples
        '''
        all_grads_test_p, all_grads_p_list, Q_list = self.TRAK_feature()

         # Initialize MIS tensor
        num_test_samples = all_grads_test_p.size(0)
        MIS = torch.zeros(num_test_samples, k, dtype=torch.int32)

        ensemble_num = all_grads_p_list.size(0)

        #Calculate average of Q
        avg_Q = torch.mean(Q_list, dim=0)

        # Calculate average of x_invXTX_XT
        x_invXTX_XT_list = []
        for l in range(ensemble_num):
            x_invXTX_XT_list.append(all_grads_test_p @ torch.linalg.inv(all_grads_p_list[l].T @ all_grads_p_list[l]) @ all_grads_p_list[l].T)
        avg_x_invXTX_XT = torch.mean(torch.stack(x_invXTX_XT_list), dim=0)

        # Compute score
        score = avg_x_invXTX_XT @ avg_Q

        # Sort and get the indices of top k influential samples for each test sample
        for i in range(num_test_samples):
            MIS[i, :] = torch.topk(score[i], k).indices

        return MIS

    def adaptive_most_k(self, k):
        all_grads_test_p, all_grads_p_list, Q_list = self.TRAK_feature()

        # Initialize MIS tensor
        num_test_samples = all_grads_test_p.size(0)
        MIS = torch.zeros(num_test_samples, k, dtype=torch.int32)

        ensemble_num = all_grads_p_list.size(0)
        train_num = all_grads_p_list.size(1)

        # Iterate over each test sample
        for j in range(num_test_samples):
            index = [i for i in range(train_num)]
            for i in range(k):
                if i == 0:
                    all_grads_p_list_cpy = all_grads_p_list.clone()
                    Q_list_cpy = Q_list.clone()
                else:
                    all_grads_p_list = []
                    Q_list = []

                    for checkpoint_id, checkpoint_file in enumerate(tqdm(self.model_checkpoints)):
                        self.model.load_state_dict(torch.load(checkpoint_file))
                        # train k more steps without the most influential sample:
                        dataset_index = [l for l in range(train_num) if l not in MIS[j, :i]]
                        new_train_loader = DataLoader(self.train_loader.dataset, batch_size=self.train_loader.batch_size, sampler=SubsetSamper(dataset_index))
                        self.model.train_with_seed(new_train_loader, epochs=3, seed=0)

                        self.model.eval()

                        parameters = list(self.model.parameters())
                        normalize_factor = torch.sqrt(torch.tensor(count_parameters(self.model), dtype=torch.float32))

                        # projection of the grads
                        # projector = CudaProjector(grad_dim=count_parameters(self.model), proj_dim=self.proj_dim, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)
                        projector = BasicProjector(grad_dim=count_parameters(self.model), proj_dim=self.proj_dim, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)

                        # Go through the training loader to get grads
                        # Φ
                        all_grads_p = grad_calculator(data_loader=self.train_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)
                        #Q
                        out_to_loss_grads = out_to_loss_grad_calculator(data_loader=self.train_loader, model=self.model, func=self.model_output_class.get_out_to_loss_grad)
                        # ϕ
                        all_grads_test_p = grad_calculator(data_loader=self.test_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)

                        # Append to list for later averaging
                        all_grads_p_list.append(all_grads_p)
                        Q_list.append(out_to_loss_grads)

                    # Convert lists to tensors
                    all_grads_p_list_cpy = torch.stack(all_grads_p_list)
                    Q_list_cpy = torch.stack(Q_list)

                # Calculate average of Q
                avg_Q = torch.mean(Q_list_cpy, dim=0)

                # Calculate average of x_invXTX_XT
                x_invXTX_XT_list = []
                for l in range(ensemble_num):
                    x_invXTX_XT_list.append(all_grads_test_p[j] @ torch.linalg.inv(all_grads_p_list_cpy[l].T @ all_grads_p_list_cpy[l]) @ all_grads_p_list_cpy[l].T)
                avg_x_invXTX_XT = torch.mean(torch.stack(x_invXTX_XT_list), dim=0)

                # Compute score
                score = avg_x_invXTX_XT @ avg_Q

                # Select the most influential sample
                max_idx = torch.topk(score, 1).indices
                MIS[j, i] = index[max_idx]

                # Remove it from the training set
                Q_list_cpy = torch.cat([torch.cat([Q_list_cpy[:, :max_idx, :max_idx], Q_list_cpy[:, :max_idx, max_idx+1:]], dim=2),torch.cat([Q_list_cpy[:, max_idx+1:, :max_idx], Q_list_cpy[:, max_idx+1:, max_idx+1:]], dim=2)],dim=1)
                all_grads_p_list_cpy = torch.cat([all_grads_p_list_cpy[:, :max_idx, :], all_grads_p_list_cpy[:, max_idx+1:, :]], dim=1)
                index = index[:max_idx] + index[max_idx + 1:]

        return MIS