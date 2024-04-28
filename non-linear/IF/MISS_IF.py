import torch
from .projector import CudaProjector, ProjectionType, BasicProjector
from .grad_calculator import count_parameters, grad_calculator, out_to_loss_grad_calculator
from tqdm import tqdm

class MISS_IF:
    def __init__(self,
                 model,
                 model_checkpoints,
                 train_loader,
                 test_loader,
                 model_output_class,
                 device):
        '''
        :param model: a nn.module instance, no need to load any checkpoint
        :param model_checkpoints: a list of checkpoint path, the length of this list indicates the ensemble model number
        :param train_loader: train samples in a data loader
        :param train_loader: test samples in a data loader
        :param model_output_class: a class definition inheriting BaseModelOutputClass
        :param device: the device running
        '''
        self.model = model
        self.model_checkpoints = model_checkpoints
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_output_class = model_output_class
        self.device = device
        assert isinstance(self.model_checkpoints[0], str), "items in model_checkpoints should be str if you state two_stage=False"

    def most_k(self, k):
        '''
        Select the most influential k samples
        '''
        running_avg_x_invXTX_XT = 0  # using broadcast
        running_counter_x_invXTX_XT = 0
        running_avg_Q = 0  # using broadcast
        running_counter_Q = 0

        for checkpoint_id, checkpoint_file in enumerate(tqdm(self.model_checkpoints)):
            self.model.load_state_dict(torch.load(checkpoint_file))
            self.model.eval()

            print(self.model)
            print("#Parameters:", count_parameters(self.model))
            parameters = list(self.model.parameters())
            normalize_factor = torch.sqrt(torch.tensor(count_parameters(self.model), dtype=torch.float32))

            # projection of the grads
            # projector = CudaProjector(grad_dim=count_parameters(self.model), proj_dim=2048, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)
            projector = BasicProjector(grad_dim=count_parameters(self.model), proj_dim=2048, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)

            # Go through the training loader to get grads

            # Φ
            all_grads_p = grad_calculator(data_loader=self.train_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)
            out_to_loss_grads = out_to_loss_grad_calculator(data_loader=self.train_loader, model=self.model, func=self.model_output_class.get_out_to_loss_grad)
            # ϕ
            all_grads_test_p = grad_calculator(data_loader=self.test_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)

            x_invXTX_XT = all_grads_test_p @ torch.linalg.inv(all_grads_p.T @ all_grads_p) @ all_grads_p.T
            Q = out_to_loss_grads

            # Use running avg to reduce mem usage
            running_avg_x_invXTX_XT = running_avg_x_invXTX_XT * running_counter_x_invXTX_XT + x_invXTX_XT.cpu().clone().detach()
            running_avg_Q = running_avg_Q * running_counter_Q + Q.cpu().clone().detach()

            running_counter_x_invXTX_XT += 1
            running_counter_Q += 1

            running_avg_x_invXTX_XT /= running_counter_x_invXTX_XT
            running_avg_Q /= running_counter_Q

        print(running_avg_Q.shape)
        print(running_avg_x_invXTX_XT.shape)
        score = running_avg_x_invXTX_XT @ running_avg_Q

         # Initialize MIS tensor
        num_test_samples = all_grads_test_p.size(0)
        MIS = torch.zeros(num_test_samples, k, dtype=torch.int32)

        # Sort and get the indices of top k influential samples for each test sample
        for i in range(num_test_samples):
            top_k_indices = score[i].cpu().detach().numpy().argsort()[-k:][::-1]
            top_k_indices_copy = top_k_indices.copy()  # Make a copy of the numpy array
            MIS[i, :] = torch.tensor(top_k_indices_copy, dtype=torch.int32)

        return MIS

    def adaptive_most_k(self, k):
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
            # projector = CudaProjector(grad_dim=count_parameters(self.model), proj_dim=2048, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)
            projector = BasicProjector(grad_dim=count_parameters(self.model), proj_dim=2048, seed=0, proj_type=ProjectionType.rademacher, device="cuda", max_batch_size=8)

            # Go through the training loader to get grads
            # Φ
            all_grads_p = grad_calculator(data_loader=self.train_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)
            out_to_loss_grads = out_to_loss_grad_calculator(data_loader=self.train_loader, model=self.model, func=self.model_output_class.get_out_to_loss_grad)
            # ϕ
            all_grads_test_p = grad_calculator(data_loader=self.test_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=projector, checkpoint_id=checkpoint_id)
            # Append to list for later averaging
            all_grads_p_list.append(all_grads_p)
            Q_list.append(out_to_loss_grads)

        # Convert lists to tensors
        all_grads_p_tensor = torch.stack(all_grads_p_list)
        Q_tensor = torch.stack(Q_list)

        # Initialize MIS tensor
        num_test_samples = all_grads_test_p.size(0)
        MIS = torch.zeros(num_test_samples, k, dtype=torch.int32)

        # Iterate over each test sample
        for j in range(num_test_samples):
            index = [i for i in range(all_grads_p_tensor.size(1))]
            for i in range(k):
                avg_Q = torch.mean(Q_tensor, dim=0)
                avg_all_grads_p = torch.mean(all_grads_p_tensor, dim=0)
                score = all_grads_test_p[j] @ torch.linalg.inv(avg_all_grads_p.T @ avg_all_grads_p) @ avg_all_grads_p.T @ avg_Q
                # Select the most influential sample
                i_max = score.cpu().detach().numpy().flatten().argsort()[-1]
                MIS[j, i] = index[i_max]

                # Remove it from the training set
                Q_tensor = torch.cat([Q_tensor[:i_max], Q_tensor[i_max + 1:]], dim=0)
                all_grads_p_tensor = torch.cat([all_grads_p_tensor[:i_max], all_grads_p_tensor[i_max + 1:]], dim=0)
                index = index[:i_max] + index[i_max + 1:]

        return MIS