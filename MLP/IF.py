import torch
from pydvl.influence.torch import EkfacInfluence
from utlis.grad_calculator import count_parameters, grad_calculator
from utlis.data import data_generation
from tqdm import tqdm
import copy

class MISS_IF:
    def __init__(self,
                 model,
                 model_checkpoints,
                 train_loader,
                 test_loader,
                 ensemble,
                 model_output_class,
                 seed,
                 device):
        '''
        :param model: a nn.module instance, no need to load any checkpoint
        :param model_checkpoints: a list of checkpoint path, the length of this list indicates the ensemble model number
        :param train_loader: train samples in a data loader
        :param test_loader: test samples in a data loader
        :param ensemble: ensemble number
        :param model_output_class: a class definition inheriting BaseModelOutputClass
        :param seed: previous experiment's seed
        :param device: the device running
        '''
        self.model = model

        self.model_checkpoints = model_checkpoints
        self.model_checkpoints_cpy = copy.deepcopy(model_checkpoints)

        self.train_loader = train_loader
        self.train_loader_cpy = copy.deepcopy(train_loader)

        self.test_loader = test_loader
        self.test_loader_cpy = copy.deepcopy(test_loader) # We won't touch this but just to be safe

        self.ensemble = ensemble

        self.model_output_class = model_output_class
        self.seed = seed
        self.device = device

    def _convert_from_loader(self, loader):
        data = [(features, labels) for features, labels in loader]
        concatenated_data = [torch.cat([item[i] for item in data], dim=0) for i in range(len(data[0]))]

        return concatenated_data

    def _reset(self):
        self.model_checkpoints = self.model_checkpoints_cpy
        self.train_loader = self.train_loader_cpy
        self.test_loader = self.test_loader_cpy

    def most_k(self, k):
        influence_sum = 0

        train_data = self._convert_from_loader(self.train_loader)

        for checkpoint_id, checkpoint_file in enumerate(self.model_checkpoints):
            self.model.load_state_dict(torch.load(checkpoint_file))
            self.model.eval()
            influence_model = EkfacInfluence(
                self.model,
                update_diagonal=True,
                hessian_regularization=0.001,
            )
            influence_model = influence_model.fit(self.train_loader)

            parameters = list(self.model.parameters())
            normalize_factor = torch.sqrt(torch.tensor(count_parameters(self.model), dtype=torch.float32))

            all_grads_test_p = grad_calculator(data_loader=self.test_loader, model=self.model, parameters=parameters, func=self.model_output_class.model_output, normalize_factor=normalize_factor, device=self.device, projector=None, checkpoint_id=checkpoint_id)

            influence_factors = influence_model.influence_factors(*train_data)
            influence_sum += (influence_factors @ all_grads_test_p.T).T

        influence_mean = influence_sum / len(self.model_checkpoints)

        # Calculate the indices of top k influential samples for each test sample
        test_size = len(self.test_loader)
        MIS = torch.zeros(test_size, k, dtype=torch.int32)

        for i in range(test_size):
            MIS[i, :] = torch.topk(influence_mean[i], k).indices

        self._reset()
        return MIS

    def adaptive_most_k(self, k, warm_start=True, step_size=5):
        test_size = len(self.test_loader)
        train_size = len(self.train_loader)
        MIS = torch.zeros(test_size, k, dtype=torch.int32)

        for i in tqdm(range(test_size)):
            index = list(range(train_size))
            step = step_size
            for j in range(0, k, step_size):
                self.train_loader, _ = data_generation([l for l in range(train_size) if l not in MIS[i, :j]], list(range(test_size)), mode='MISS')

                # handle overflow
                if j + step > k:
                    step = k - j

                max_idx_list = self.most_k(step)[i, :]
                MIS[i, j:j+step] = torch.tensor([index[l] for l in max_idx_list])
                index = [index[i] for i in range(len(index)) if i not in max_idx_list]

                # update the model, the dataset
                train_loader, _ = data_generation([l for l in range(train_size) if l not in MIS[i, :j+step]], list(range(test_size)), mode='train')

                for idx, checkpoint_file in enumerate(self.model_checkpoints):
                    if warm_start:
                        self.model.load_state_dict(torch.load(checkpoint_file))
                        self.model.train_with_seed(train_loader, epochs=8, seed=idx, verbose=False)
                        torch.save(self.model.state_dict(), f"./checkpoint/adaptive_tmp/seed_{self.seed}_ensemble_{idx}_w.pt")
                    else:
                        self.model = self.model_cpy
                        self.model.train_with_seed(train_loader, epochs=30, seed=idx, verbose=False)
                        torch.save(self.model.state_dict(), f"./checkpoint/adaptive_tmp/seed_{self.seed}_ensemble_{idx}.pt")

                if warm_start:
                    self.model_checkpoints = [f"./checkpoint/adaptive_tmp/seed_{self.seed}_ensemble_{idx}_w.pt" for idx in range(self.ensemble)]
                else:
                    self.model_checkpoints = [f"./checkpoint/adaptive_tmp/seed_{self.seed}_ensemble_{idx}.pt" for idx in range(self.ensemble)]
            self._reset()

        return MIS