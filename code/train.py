import os
import random

import numpy as np
import torch
import torch.distributions as D
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from datasets.argoverse.dataset import ArgoH5Dataset
from datasets.interaction_dataset.dataset import InteractionDataset
from datasets.nuscenes.dataset import NuscenesH5Dataset
from datasets.synth.dataset import SynthV1CausalDataset, my_collate_fn
from datasets.trajnetpp.dataset import TrajNetPPDataset
from models.autobot_ego import AutoBotEgo
from models.autobot_joint import AutoBotJoint
from process_args import get_train_args
from utils.metric_helpers import min_xde_K
from utils.train_helpers import nll_loss_multimodes, nll_loss_multimodes_joint, calc_consistency_loss, HNC_ARS, calc_contrastive_loss, calc_ranking_loss, ACEs
import wandb
import pickle


class Trainer:
    def __init__(self, args, results_dirname):
        self.args = args
        self.results_dirname = results_dirname
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available() and not self.args.disable_cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.args.seed)
        else:
            self.device = torch.device("cpu")

        self.initialize_dataloaders()
        self.initialize_model()
        self.optimiser = optim.Adam(self.autobot_model.parameters(), lr=self.args.learning_rate,
                                    eps=self.args.adam_epsilon)

        if self.args.weight_path != "":
            self.load_optimiser()
            new_learning_rate = self.args.learning_rate
            for milestone in self.args.learning_rate_sched:
                if self.args.start_epoch >= milestone:
                    new_learning_rate *= 0.5
            for param_group in self.optimiser.param_groups:
                param_group['lr'] = new_learning_rate

        self.optimiser_scheduler = MultiStepLR(self.optimiser, milestones=args.learning_rate_sched, gamma=0.5,
                                               verbose=True, last_epoch=self.args.start_epoch if self.args.weight_path != "" else -1)
        if self.args.reg_type == "consistency":
            encoder_params = list(self.autobot_model.social_attn_layers.parameters()) + \
                             list(self.autobot_model.temporal_attn_layers.parameters()) #+ \
            self.consistency_optimiser = optim.Adam(encoder_params, lr=self.optimiser.param_groups[0]["lr"],
                                                    eps=self.args.adam_epsilon)
            self.consistency_scheduler = MultiStepLR(self.consistency_optimiser, milestones=args.learning_rate_sched, gamma=0.5,
                                                     verbose=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.results_dirname, "tb_files"))

        self.smallest_minade_k = 5.0  # for computing best models
        self.smallest_minfde_k = 5.0  # for computing best models

    def initialize_dataloaders(self):
        if "Nuscenes" in self.args.dataset:
            train_dset = NuscenesH5Dataset(dset_path=self.args.dataset_path, split_name="train",
                                           model_type=self.args.model_type, use_map_img=self.args.use_map_image,
                                           use_map_lanes=self.args.use_map_lanes)
            val_dset = NuscenesH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                         model_type=self.args.model_type, use_map_img=self.args.use_map_image,
                                         use_map_lanes=self.args.use_map_lanes)

        elif "interaction-dataset" in self.args.dataset:
            train_dset = InteractionDataset(dset_path=self.args.dataset_path, split_name="train",
                                            use_map_lanes=self.args.use_map_lanes, evaluation=False)
            val_dset = InteractionDataset(dset_path=self.args.dataset_path, split_name="val",
                                          use_map_lanes=self.args.use_map_lanes, evaluation=False)

        elif "trajnet++" in self.args.dataset:
            train_dset = TrajNetPPDataset(dset_path=self.args.dataset_path, split_name="train")
            val_dset = TrajNetPPDataset(dset_path=self.args.dataset_path, split_name="test")

        elif "Argoverse" in self.args.dataset:
            train_dset = ArgoH5Dataset(dset_path=self.args.dataset_path, split_name="train",
                                       use_map_lanes=self.args.use_map_lanes)
            val_dset = ArgoH5Dataset(dset_path=self.args.dataset_path, split_name="val",
                                     use_map_lanes=self.args.use_map_lanes)
        elif self.args.dataset == "synth":
            train_dset = SynthV1CausalDataset(dset_path=self.args.dataset_path, split="train", size=self.args.train_data_size)
            val_dset = SynthV1CausalDataset(dset_path=self.args.dataset_path, split="val")
        elif self.args.dataset == 's2r':
            # real and sim datasets
            train_dset_real = TrajNetPPDataset(dset_path=self.args.dataset_path_real, split_name="train", proportion=self.args.low_data)
            val_dset_real = TrajNetPPDataset(dset_path=self.args.dataset_path_real, split_name="test")
            train_dset_sim = SynthV1CausalDataset(dset_path=self.args.dataset_path_synth, split="train", size=self.args.train_data_size) 
            val_dset_sim = SynthV1CausalDataset(dset_path=self.args.dataset_path_synth, split="val")
        else:
            raise NotImplementedError

        if self.args.dataset == 's2r':
            self.num_other_agents = train_dset_real.num_others
            self.pred_horizon = train_dset_real.pred_horizon
            self.k_attr = train_dset_real.k_attr
            self.map_attr = train_dset_real.map_attr
            self.predict_yaw = train_dset_real.predict_yaw
        else:
            self.num_other_agents = train_dset.num_others
            self.pred_horizon = train_dset.pred_horizon
            self.k_attr = train_dset.k_attr
            self.map_attr = train_dset.map_attr
            self.predict_yaw = train_dset.predict_yaw
        
        if "Joint" in self.args.model_type:
            self.num_agent_types = train_dset.num_agent_types

        if self.args.dataset == "synth":
            self.train_loader = torch.utils.data.DataLoader(
                train_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False,
                pin_memory=False, collate_fn=my_collate_fn
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dset, batch_size=512, shuffle=True, num_workers=12, drop_last=False,
                pin_memory=False, collate_fn=my_collate_fn
            )
        elif self.args.dataset == "s2r":
            # Real
            self.train_loader_real = torch.utils.data.DataLoader(
                train_dset_real, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dset_real, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
            )
            # Sim
            self.train_loader_sim = torch.utils.data.DataLoader(
                train_dset_sim, batch_size=self.args.batch_size_sim, shuffle=True, num_workers=12, drop_last=False,
                pin_memory=False, collate_fn=my_collate_fn
            )
            self.val_loader_sim = torch.utils.data.DataLoader(
                val_dset_sim, batch_size=self.args.batch_size_sim, shuffle=True, num_workers=12, drop_last=False,
                pin_memory=False, collate_fn=my_collate_fn
            )
        else:
            self.train_loader = torch.utils.data.DataLoader(
                train_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
            )
            self.val_loader = torch.utils.data.DataLoader(
                val_dset, batch_size=self.args.batch_size, shuffle=True, num_workers=12, drop_last=False, pin_memory=False
            )
        
        if self.args.dataset == "s2r":
            print("Train dataset loaded with length", len(train_dset_real))
            print("Val dataset loaded with length", len(val_dset_real)) 
        else:
            print("Train dataset loaded with length", len(train_dset))
            print("Val dataset loaded with length", len(val_dset))

    def initialize_model(self):
        if "Ego" in self.args.model_type:
            self.autobot_model = AutoBotEgo(k_attr=self.k_attr,
                                            d_k=self.args.hidden_size,
                                            _M=self.num_other_agents,
                                            c=self.args.num_modes,
                                            T=self.pred_horizon,
                                            L_enc=self.args.num_encoder_layers,
                                            dropout=self.args.dropout,
                                            num_heads=self.args.tx_num_heads,
                                            L_dec=self.args.num_decoder_layers,
                                            tx_hidden_size=self.args.tx_hidden_size,
                                            use_map_img=self.args.use_map_image,
                                            use_map_lanes=self.args.use_map_lanes,
                                            map_attr=self.map_attr,
                                            return_embeddings=((self.args.reg_type in ["contrastive", "ranking"] and self.args.dataset == "synth") or self.args.dataset == "s2r"),
                                            projector_dim1=self.args.projector_dim1,
                                            projector_dim2=self.args.projector_dim2,
                                            projector_dim3=self.args.projector_dim3).to(self.device)

        elif "Joint" in self.args.model_type:
            self.autobot_model = AutoBotJoint(k_attr=self.k_attr,
                                              d_k=self.args.hidden_size,
                                              _M=self.num_other_agents,
                                              c=self.args.num_modes,
                                              T=self.pred_horizon,
                                              L_enc=self.args.num_encoder_layers,
                                              dropout=self.args.dropout,
                                              num_heads=self.args.tx_num_heads,
                                              L_dec=self.args.num_decoder_layers,
                                              tx_hidden_size=self.args.tx_hidden_size,
                                              use_map_lanes=self.args.use_map_lanes,
                                              map_attr=self.map_attr,
                                              num_agent_types=self.num_agent_types,
                                              predict_yaw=self.predict_yaw).to(self.device)
        else:
            raise NotImplementedError

        if self.args.weight_path != "":
            self.load_model()

    def _data_to_device(self, data, model_type_overwrite=None):
        model_type = self.args.model_type
        if model_type_overwrite is not None:
            model_type = model_type_overwrite

        if "Joint" in model_type:
            ego_in, ego_out, agents_in, agents_out, context_img, agent_types = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            agents_out = agents_out.float().to(self.device)
            context_img = context_img.float().to(self.device)
            agent_types = agent_types.float().to(self.device)
            return ego_in, ego_out, agents_in, agents_out, context_img, agent_types

        elif "Ego" in model_type:
            ego_in, ego_out, agents_in, roads = data
            ego_in = ego_in.float().to(self.device)
            ego_out = ego_out.float().to(self.device)
            agents_in = agents_in.float().to(self.device)
            roads = roads.float().to(self.device)
            return ego_in, ego_out, agents_in, roads

    def _compute_ego_errors(self, ego_preds, ego_gt):
        ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
        ade_losses = torch.mean(torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1).transpose(0, 1).cpu().numpy()
        fde_losses = torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1).transpose(0, 1).cpu().numpy()
        return ade_losses, fde_losses

    def _compute_marginal_errors(self, preds, ego_gt, agents_gt, agents_in):
        agent_masks = torch.cat((torch.ones((len(agents_in), 1)).to(self.device), agents_in[:, -1, :, -1]), dim=-1).view(1, 1, len(agents_in), -1)
        agent_masks[agent_masks == 0] = float('nan')
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2).unsqueeze(0).permute(0, 2, 1, 3, 4)
        error = torch.norm(preds[:, :, :, :, :2] - agents_gt[:, :, :, :, :2], 2, dim=-1) * agent_masks
        ade_losses = np.nanmean(error.cpu().numpy(), axis=1).transpose(1, 2, 0)
        fde_losses = error[:, -1].cpu().numpy().transpose(1, 2, 0)
        return ade_losses, fde_losses

    def _compute_joint_errors(self, preds, ego_gt, agents_gt):
        agents_gt = torch.cat((ego_gt.unsqueeze(2), agents_gt), dim=2)
        agents_masks = agents_gt[:, :, :, -1]
        agents_masks[agents_masks == 0] = float('nan')
        ade_losses = []
        for k in range(self.args.num_modes):
            ade_error = (torch.norm(preds[k, :, :, :, :2].transpose(0, 1) - agents_gt[:, :, :, :2], 2, dim=-1)
                         * agents_masks).cpu().numpy()
            ade_error = np.nanmean(ade_error, axis=(1, 2))
            ade_losses.append(ade_error)
        ade_losses = np.array(ade_losses).transpose()

        fde_losses = []
        for k in range(self.args.num_modes):
            fde_error = (torch.norm(preds[k, -1, :, :, :2] - agents_gt[:, -1, :, :2], 2, dim=-1) * agents_masks[:, -1]).cpu().numpy()
            fde_error = np.nanmean(fde_error, axis=1)
            fde_losses.append(fde_error)
        fde_losses = np.array(fde_losses).transpose()

        return ade_losses, fde_losses

    def autobotego_train(self):
        steps = 0
        if self.args.reg_type in ["ranking", "contrastive"]:
            print("Freezing the model except the contrastive projector")
            for param in self.autobot_model.parameters():
                param.requires_grad = False
            for param in self.autobot_model.contrastive_projector.parameters():
                param.requires_grad = True
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            print("Epoch:", epoch)
            if self.args.reg_type in ["contrastive", "ranking"] and epoch - self.args.start_epoch == self.args.num_proj_warmup_epochs:
                print("Unfreezing the model")
                for param in self.autobot_model.parameters():
                    param.requires_grad = True
            # adjusting the learning rate for ranking
            if self.args.reg_type == "ranking" and self.args.dataset != "s2r":
                for param_group in self.optimiser.param_groups:
                    param_group['lr'] *= 10
            epoch_ade_losses = []
            epoch_fde_losses = []
            epoch_mode_probs = []

            if self.args.dataset == "s2r":
                # iter
                self.iter_train_loader_real = iter(self.train_loader_real)
                self.iter_train_loader_sim = iter(self.train_loader_sim)
                # zip 
                self.train_loader = zip(self.iter_train_loader_real, self.iter_train_loader_sim)

            for i, data in enumerate(self.train_loader):
                if self.args.dataset == "synth":
                    scenes, causal_effects, directly_causals, data_splits = data
                    causal_effects = [torch.Tensor(causal_effect).float().to(self.device) for causal_effect in causal_effects]
                    directly_causals = [torch.Tensor(directly_causal).bool().to(self.device) for directly_causal in directly_causals]
                    if self.args.reg_type == "None":
                        scenes = [data[data_splits[:-1]] for data in scenes]
                    elif self.args.reg_type == "augment":
                        mask = np.zeros(len(scenes[0]))
                        mask[data_splits[:-1]] = 1
                        for sample_id in range(len(causal_effects)):
                            mask[data_splits[sample_id] + 1:data_splits[sample_id + 1]] = causal_effects[sample_id].cpu().numpy() <= 0.02
                        mask = torch.tensor(mask).bool()
                        scenes = [data[mask] for data in scenes]
                    ego_in, ego_out, agents_in, _, context_img, _ = self._data_to_device(scenes, "Joint")
                    roads = context_img
                elif "trajnet++" in self.args.dataset:
                    ego_in, ego_out, agents_in, context_img = self._data_to_device(data)
                    roads = context_img
                elif self.args.dataset == "s2r":
                    data_real, data_sim = data
                    # sim 
                    scenes, causal_effects, directly_causals, data_splits = data_sim 
                    causal_effects = [torch.Tensor(causal_effect).float().to(self.device) for causal_effect in causal_effects]
                    directly_causals = [torch.Tensor(directly_causal).bool().to(self.device) for directly_causal in directly_causals]
                    # augmentation
                    if self.args.reg_type == "augment":
                        mask = np.zeros(len(scenes[0]))
                        mask[data_splits[:-1]] = 1
                        for sample_id in range(len(causal_effects)):
                            mask[data_splits[sample_id] + 1:data_splits[sample_id + 1]] = causal_effects[sample_id].cpu().numpy() <= 0.02
                        mask = torch.tensor(mask).bool()
                        scenes = [data[mask] for data in scenes]

                    ego_in_sim, ego_out_sim, agents_in_sim, _, context_img, _ = self._data_to_device(scenes, "Joint")
                    roads_sim = context_img
                   
                    # real
                    ego_in_real, ego_out_real, agents_in_real, roads_real = self._data_to_device(data_real)
                else:
                    ego_in, ego_out, agents_in, roads = self._data_to_device(data)

                if self.args.dataset == "synth" and self.args.reg_type in ["contrastive", "ranking"]:
                    pred_obs, mode_probs, embeds = self.autobot_model(ego_in, agents_in, roads)
                elif self.args.dataset == "s2r":
                    # Forward 2 times
                    # Real
                    pred_obs_real, mode_probs_real, embeds_real = self.autobot_model(ego_in_real, agents_in_real, roads_real)
                    # Sim
                    pred_obs_sim, mode_probs_sim, embeds_sim = self.autobot_model(ego_in_sim, agents_in_sim, roads_sim)
                else:
                    pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads)

                if self.args.dataset == "synth" and self.args.reg_type in ["consistency", "contrastive", "ranking"]:
                    nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(
                        pred_obs[:, :, data_splits[:-1], :], ego_out[data_splits[:-1], :, :2],
                        mode_probs[data_splits[:-1]],
                        entropy_weight=self.args.entropy_weight,
                        kl_weight=self.args.kl_weight,
                        use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)
                elif self.args.dataset == "s2r":
                    if self.args.reg_type == "baseline" or self.args.reg_type == "augment":
                        # Pred real and sim
                        # Real
                        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs_real, ego_out_real[:, :, :2], mode_probs_real,
                                                                                        entropy_weight=self.args.entropy_weight,
                                                                                        kl_weight=self.args.kl_weight,
                                                                                        use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)
                        # Sim
                        nll_loss_sim, kl_loss_sim, post_entropy_sim, adefde_loss_sim = nll_loss_multimodes(pred_obs_sim, ego_out_sim[:, :, :2], mode_probs_sim,
                                                                                        entropy_weight=self.args.entropy_weight,
                                                                                        kl_weight=self.args.kl_weight,
                                                                                        use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)
                    else:
                        # Real
                        nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs_real, ego_out_real[:, :, :2], mode_probs_real,
                                                                                        entropy_weight=self.args.entropy_weight,
                                                                                        kl_weight=self.args.kl_weight,
                                                                                        use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)
                else:
                    nll_loss, kl_loss, post_entropy, adefde_loss = nll_loss_multimodes(pred_obs, ego_out[:, :, :2], mode_probs,
                                                                                       entropy_weight=self.args.entropy_weight,
                                                                                       kl_weight=self.args.kl_weight,
                                                                                       use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss)

                if self.args.dataset == "synth" and self.args.reg_type == "consistency":
                    consistency_loss = calc_consistency_loss(pred_obs, causal_effects, data_splits, self.args.consistency_weight)

                elif self.args.dataset == "synth" and self.args.reg_type == "contrastive":
                    contrastive_loss, poisoned_prop = calc_contrastive_loss(embeds, causal_effects, directly_causals, data_splits, self.args.contrastive_weight, poison_prob=self.args.poison_prob)

                elif self.args.dataset == "synth" and self.args.reg_type == "ranking":
                    ranking_loss, ranking_accuracy = calc_ranking_loss(embeds, causal_effects, directly_causals, data_splits, self.args.ranking_weight, self.args.ranking_margin, self.args.do_consecutive, poison_prob=self.args.poison_prob)
                
                elif self.args.dataset == "s2r" and self.args.reg_type == "contrastive":
                    contrastive_loss, poisoned_prop = calc_contrastive_loss(embeds_sim, causal_effects, directly_causals, data_splits, self.args.contrastive_weight, poison_prob=self.args.poison_prob)

                elif self.args.dataset == "s2r" and self.args.reg_type == "ranking":
                    ranking_loss, ranking_accuracy = calc_ranking_loss(embeds_sim, causal_effects, directly_causals, data_splits, self.args.ranking_weight, self.args.ranking_margin, self.args.do_consecutive, poison_prob=self.args.poison_prob)
                

                self.optimiser.zero_grad()
                if self.args.dataset == "synth" and self.args.reg_type == "consistency":
                    (nll_loss + adefde_loss + kl_loss).backward(retain_graph=True)
                elif self.args.dataset == "synth" and self.args.reg_type == "contrastive":
                    (nll_loss + adefde_loss + kl_loss + contrastive_loss).backward()
                elif self.args.dataset == "synth" and self.args.reg_type == "ranking":
                    (nll_loss + adefde_loss + kl_loss + ranking_loss).backward()
                elif self.args.dataset == "s2r" and self.args.reg_type == "baseline":
                    ((nll_loss + adefde_loss + kl_loss) + (nll_loss_sim + adefde_loss_sim + kl_loss_sim)).backward()
                elif self.args.dataset == "s2r" and self.args.reg_type == "augment":
                    ((nll_loss + adefde_loss + kl_loss) + (nll_loss_sim + adefde_loss_sim + kl_loss_sim)).backward()
                elif self.args.dataset == "s2r" and self.args.reg_type == "contrastive":
                    (nll_loss + adefde_loss + kl_loss + contrastive_loss).backward()
                elif self.args.dataset == "s2r" and self.args.reg_type == "ranking":
                    (nll_loss + adefde_loss + kl_loss + ranking_loss).backward()
                else:
                    (nll_loss + adefde_loss + kl_loss).backward()

                nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
                self.optimiser.step()

                if self.args.dataset == "synth" and self.args.reg_type == "consistency":
                    self.consistency_optimiser.zero_grad()
                    consistency_loss.backward()
                    nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
                    self.consistency_optimiser.step()

               
                self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)
                if self.args.dataset == "synth" and self.args.reg_type == "consistency":
                    self.writer.add_scalar("Loss/consistency", consistency_loss.item(), steps)
                elif self.args.dataset == "synth" and self.args.reg_type == "contrastive":
                    self.writer.add_scalar("Loss/contrastive", contrastive_loss.item(), steps)
                    wandb.log({"Loss/nll": nll_loss.item(), "Loss/adefde": adefde_loss.item(), "contrastive_loss": contrastive_loss.item() / self.args.contrastive_weight, "epoch": epoch, "batch": i, "poisoned_prop": poisoned_prop})
                elif self.args.dataset == "synth" and self.args.reg_type == "ranking":
                    self.writer.add_scalar("Loss/ranking", ranking_loss.item(), steps)
                    wandb.log({"Loss/nll": nll_loss.item(), "Loss/adefde": adefde_loss.item(),
                               "Loss/ranking_loss": ranking_loss.item() / self.args.ranking_weight * 1000, "epoch": epoch, "ranking accuracy": ranking_accuracy})
                elif self.args.dataset == "s2r" and self.args.reg_type == "contrastive":
                    self.writer.add_scalar("Loss/contrastive", contrastive_loss.item(), steps)
                elif self.args.dataset == "s2r" and self.args.reg_type == "ranking":
                    self.writer.add_scalar("Loss/ranking", ranking_loss.item(), steps)

                with torch.no_grad():
                    if self.args.dataset == "s2r":
                        ade_losses, fde_losses = self._compute_ego_errors(pred_obs_real, ego_out_real)
                    else:
                        ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                    epoch_ade_losses.append(ade_losses)
                    epoch_fde_losses.append(fde_losses)
                    if self.args.dataset == "s2r":
                        epoch_mode_probs.append(mode_probs_real.detach().cpu().numpy())
                    else:
                        epoch_mode_probs.append(mode_probs.detach().cpu().numpy())

                # Learning curves for steps
                if self.args.dataset == "s2r" and steps % 10 == 0:
                    #get train ADE for each step here
                    ade_losses_step = epoch_ade_losses[-1]
                    fde_losses_step = epoch_fde_losses[-1]
                    mode_probs_step = epoch_mode_probs[-1]

                    train_minade_c_step = min_xde_K(ade_losses_step, mode_probs_step, K=self.args.num_modes)
                    train_minade_10_step = min_xde_K(ade_losses_step, mode_probs_step, K=min(self.args.num_modes, 10))
                    train_minade_5_step = min_xde_K(ade_losses_step, mode_probs_step, K=min(self.args.num_modes, 5))
                    train_minade_1_step = min_xde_K(ade_losses_step, mode_probs_step, K=1)
                    train_minfde_c_step = min_xde_K(fde_losses_step, mode_probs_step, K=min(self.args.num_modes, 10))
                    train_minfde_1_step = min_xde_K(fde_losses_step, mode_probs_step, K=1)
                    print("Train minADE c:", train_minade_c_step[0], "Train minADE 1:", train_minade_1_step[0], "Train minFDE c:", train_minfde_c_step[0])

                    # Log train metrics
                    self.writer.add_scalar("metrics step/Train minADE_{}".format(self.args.num_modes), train_minade_c_step[0], steps)
                    self.writer.add_scalar("metrics step/Train minADE_{}".format(10), train_minade_10_step[0], steps)
                    self.writer.add_scalar("metrics step/Train minADE_{}".format(5), train_minade_5_step[0], steps)
                    self.writer.add_scalar("metrics step/Train minADE_{}".format(1), train_minade_1_step[0], steps)
                    self.writer.add_scalar("metrics step/Train minFDE_{}".format(self.args.num_modes), train_minfde_c_step[0], steps)
                    self.writer.add_scalar("metrics step/Train minFDE_{}".format(1), train_minfde_1_step[0], steps)

                    # get val ADE for each step here
                    self.autobotego_evaluate_step(steps)

                if i % 10 == 0:
                    if self.args.dataset == "synth" and self.args.reg_type == "consistency":
                        print(i, "/", len(self.train_loader.dataset)//self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2),
                              "Consistency loss", round(consistency_loss.item(), 2))
                    elif self.args.dataset == "synth" and self.args.reg_type == "contrastive":
                        print(i, "/", len(self.train_loader.dataset) // self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2),
                              "Contrastive loss", round(contrastive_loss.item(), 2))
                    elif self.args.dataset == "synth" and self.args.reg_type == "ranking":
                        print(i, "/", len(self.train_loader.dataset) // self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2),
                              "Ranking loss", round(ranking_loss.item(), 2))
                    elif self.args.dataset == "s2r" and self.args.reg_type == "contrastive":
                        print(i, "/", len(self.train_loader_real.dataset) // self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs_real).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2),
                              "Contrastive loss", round(contrastive_loss.item(), 2))
                    elif self.args.dataset == "s2r" and self.args.reg_type == "ranking":
                        print(i, "/", len(self.train_loader_real.dataset) // self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs_real).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2),
                              "Ranking loss", round(ranking_loss.item(), 2))
                    elif self.args.dataset == "s2r" and (self.args.reg_type == "baseline" or self.args.reg_type == "augment"):
                        print(i, "/", len(self.train_loader_real.dataset) // self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs_real).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))
                    else:
                        print(i, "/", len(self.train_loader.dataset) // self.args.batch_size,
                              "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                              "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                              "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))
                steps += 1

            ade_losses = np.concatenate(epoch_ade_losses)
            fde_losses = np.concatenate(epoch_fde_losses)
            mode_probs = np.concatenate(epoch_mode_probs)

            train_minade_c = min_xde_K(ade_losses, mode_probs, K=self.args.num_modes)
            train_minade_10 = min_xde_K(ade_losses, mode_probs, K=min(self.args.num_modes, 10))
            train_minade_5 = min_xde_K(ade_losses, mode_probs, K=min(self.args.num_modes, 5))
            train_minade_1 = min_xde_K(ade_losses, mode_probs, K=1)
            train_minfde_c = min_xde_K(fde_losses, mode_probs, K=min(self.args.num_modes, 10))
            train_minfde_1 = min_xde_K(fde_losses, mode_probs, K=1)
            print("Train minADE c:", train_minade_c[0], "Train minADE 1:", train_minade_1[0], "Train minFDE c:", train_minfde_c[0])

            # Log train metrics
            # self.writer.add_scalar("metrics/Train minADE_{}".format(self.args.num_modes), train_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(10), train_minade_10[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(5), train_minade_5[0], epoch)
            self.writer.add_scalar("metrics/Train minADE_{}".format(1), train_minade_1[0], epoch)
            # self.writer.add_scalar("metrics/Train minFDE_{}".format(self.args.num_modes), train_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Train minFDE_{}".format(1), train_minfde_1[0], epoch)

            # update learning rate
            self.optimiser_scheduler.step()
            if self.args.dataset == "synth" and self.args.reg_type == "consistency":
                self.consistency_scheduler.step()

            if (epoch + 1) % self.args.val_every == 0:
                self.autobotego_evaluate(epoch)
            self.save_model(epoch + 1)
            print("Best minADE c", self.smallest_minade_k, "Best minFDE c", self.smallest_minfde_k)

    def autobotego_evaluate(self, epoch):
        self.autobot_model.eval()
        with torch.no_grad():
            val_ade_losses = []
            val_fde_losses = []
            val_mode_probs = []
            if self.args.evaluate_causal:
                val_consistency = []
                val_HNC, val_ARS = 0, []
                NC_ACEs, IC_ACEs, DC_ACEs, Ignored_ACEs = [], [], [], []
            for i, data in enumerate(self.val_loader):
                if self.args.dataset == "synth":
                    scenes, causal_effects, directly_causals, data_splits = data
                    if not self.args.evaluate_causal:
                        scenes = [data[data_splits[:-1]] for data in scenes]
                    ego_in, ego_out, agents_in, _, context_img, _ = self._data_to_device(scenes, "Joint")
                    roads = context_img
                    causal_effects = [torch.Tensor(causal_effect).float().to(self.device) for causal_effect in causal_effects]
                    directly_causals = [torch.Tensor(directly_causal).bool().to(self.device) for directly_causal in directly_causals]
                elif  "trajnet++" in self.args.dataset:
                    ego_in, ego_out, agents_in, context_img = self._data_to_device(data)
                    roads = context_img
                else:
                    ego_in, ego_out, agents_in, roads = self._data_to_device(data)

                # encode observations
                if (self.args.dataset == "synth" and self.args.reg_type in ["contrastive", "ranking"]) or self.args.dataset == "s2r":
                    pred_obs, mode_probs, _ = self.autobot_model(ego_in, agents_in, roads)
                else:
                    pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads)

                if self.args.evaluate_causal:
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:, :, data_splits[:-1], :], ego_out[data_splits[:-1], :, :2])
                    consistency_loss = calc_consistency_loss(pred_obs, causal_effects, data_splits, self.args.consistency_weight)
                    batch_HNC, batch_ARS = HNC_ARS(pred_obs, causal_effects, data_splits)
                    NC_ACE, IC_ACE, DC_ACE, Ignored_ACE = ACEs(pred_obs, causal_effects, directly_causals, data_splits)
                else:
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                val_ade_losses.append(ade_losses)
                val_fde_losses.append(fde_losses)
                if self.args.evaluate_causal:
                    val_mode_probs.append(mode_probs[data_splits[:-1]].detach().cpu().numpy())
                    val_consistency.append(consistency_loss.item())
                    val_HNC += batch_HNC
                    val_ARS += batch_ARS
                    NC_ACEs.append(NC_ACE)
                    IC_ACEs.append(IC_ACE)
                    DC_ACEs.append(DC_ACE)
                    Ignored_ACEs.append(Ignored_ACE)
                else:
                    val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_ade_losses = np.concatenate(val_ade_losses)
            val_fde_losses = np.concatenate(val_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)
            if self.args.evaluate_causal:
                val_ARS = np.concatenate(val_ARS).mean()
                NC_ACEs = torch.concatenate(NC_ACEs).cpu().numpy()
                IC_ACEs = torch.concatenate(IC_ACEs).cpu().numpy()
                DC_ACEs = torch.concatenate(DC_ACEs).cpu().numpy()
                Ignored_ACEs = torch.concatenate(Ignored_ACEs).cpu().numpy()

                def calculate_uniform_ACE(ACE):
                    bin_edges = np.linspace(0.1, 2.1, 20)
                    num_bins = 19
                    ACE_bins = []
                    for i in range(num_bins):
                        bin_mask = (ACE[:, 1] >= bin_edges[i]) & (ACE[:, 1] < bin_edges[i + 1])
                        ACE_bins.append(ACE[bin_mask, 0].mean())
                    return np.array(ACE_bins).mean()

                print("NC_ACE:", NC_ACEs[:, 0].mean(), "DC_ACE:", calculate_uniform_ACE(DC_ACEs), "IC_ICE:",
                      calculate_uniform_ACE(IC_ACEs), "Ignored_ACE:", Ignored_ACEs[:, 0].mean(), "ACE:", np.concatenate([NC_ACEs, DC_ACEs, IC_ACEs, Ignored_ACEs], axis=0)[:, 0].mean())

                with open(os.path.join(self.results_dirname, "ACEs.pkl"), "wb") as f:
                    pickle.dump((NC_ACEs, IC_ACEs, DC_ACEs, Ignored_ACEs), f)
            val_minade_c = min_xde_K(val_ade_losses, val_mode_probs, K=self.args.num_modes)
            val_minade_10 = min_xde_K(val_ade_losses, val_mode_probs, K=min(self.args.num_modes, 10))
            val_minade_5 = min_xde_K(val_ade_losses, val_mode_probs, K=5)
            val_minade_1 = min_xde_K(val_ade_losses, val_mode_probs, K=1)
            val_minfde_c = min_xde_K(val_fde_losses, val_mode_probs, K=self.args.num_modes)
            val_minfde_1 = min_xde_K(val_fde_losses, val_mode_probs, K=1)

            # Log val metrics
            # self.writer.add_scalar("metrics/Val minADE_{}".format(self.args.num_modes), val_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(10), val_minade_10[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(5), val_minade_5[0], epoch)
            self.writer.add_scalar("metrics/Val minADE_{}".format(1), val_minade_1[0], epoch)
            # self.writer.add_scalar("metrics/Val minFDE_{}".format(self.args.num_modes), val_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Val minFDE_{}".format(1), val_minfde_1[0], epoch)
            if self.args.evaluate_causal:
                self.writer.add_scalar("metrics/Val consistency", np.array(val_consistency).mean(), epoch)
                self.writer.add_scalar("metrics/Val HNC", val_HNC, epoch)
                self.writer.add_scalar("metrics/Val ARS", val_ARS, epoch)

                print("minADE c:", val_minade_c[0], "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                      "minFDE c:", val_minfde_c[0], "minFDE_1:", val_minfde_1[0], "Consistency:",
                      round(np.array(val_consistency).mean(), 2), "HNC:", val_HNC, "ARS:", round(val_ARS, 2))
                wandb.log({"NC_ACE": NC_ACEs[:, 0].mean(), "DC_ACE": calculate_uniform_ACE(DC_ACEs),
                           "IC_ACE": calculate_uniform_ACE(IC_ACEs), "Ignored_ACE": Ignored_ACEs[:, 0].mean(),
                           "ACE": np.concatenate([NC_ACEs, DC_ACEs, IC_ACEs, Ignored_ACEs], axis=0)[:, 0].mean(),
                           "epoch": epoch, "minADE": val_minade_c[0], "minFDE": val_minfde_1[0]})
            else:
                print("minADE c:", val_minade_c[0], "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                      "minFDE c:", val_minfde_c[0], "minFDE_1:", val_minfde_1[0])
            self.autobot_model.train()
            self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0])

    def autobotego_evaluate_step(self, step):
        self.autobot_model.eval()
        with torch.no_grad():
            val_ade_losses = []
            val_fde_losses = []
            val_mode_probs = []
            if self.args.evaluate_causal:
                val_consistency = []
                val_HNC, val_ARS = 0, []

            for i, data in enumerate(self.val_loader):
                if self.args.dataset == "synth":
                    scenes, causal_effects, data_splits = data
                    if not self.args.evaluate_causal:
                        scenes = [data[data_splits[:-1]] for data in scenes]
                    ego_in, ego_out, agents_in, _, context_img, _ = self._data_to_device(scenes, "Joint")
                    roads = context_img
                    causal_effects = [torch.Tensor(causal_effect).float().to(self.device) for causal_effect in causal_effects]
                elif  "trajnet++" in self.args.dataset:
                    ego_in, ego_out, agents_in, _, context_img, _ = self._data_to_device(data, "Joint")
                    roads = context_img
                else:
                    ego_in, ego_out, agents_in, roads = self._data_to_device(data)

                # encode observations
                if (self.args.dataset == "synth" and self.args.reg_type in ["contrastive", "ranking"]) or self.args.dataset == "s2r":
                    pred_obs, mode_probs, _ = self.autobot_model(ego_in, agents_in, roads)
                else:
                    pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, roads)

                if self.args.evaluate_causal:
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs[:, :, data_splits[:-1], :], ego_out[data_splits[:-1], :, :2])
                    consistency_loss = calc_consistency_loss(pred_obs, causal_effects, data_splits, self.args.consistency_weight)
                    batch_HNC, batch_ARS = HNC_ARS(pred_obs, causal_effects, data_splits)
                else:
                    ade_losses, fde_losses = self._compute_ego_errors(pred_obs, ego_out)
                val_ade_losses.append(ade_losses)
                val_fde_losses.append(fde_losses)
                if self.args.evaluate_causal:
                    val_mode_probs.append(mode_probs[data_splits[:-1]].detach().cpu().numpy())
                    val_consistency.append(consistency_loss.item())
                    val_HNC += batch_HNC
                    val_ARS += batch_ARS
                else:
                    val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_ade_losses = np.concatenate(val_ade_losses)
            val_fde_losses = np.concatenate(val_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)
            if self.args.evaluate_causal:
                val_ARS = np.concatenate(val_ARS).mean()

            val_minade_c = min_xde_K(val_ade_losses, val_mode_probs, K=self.args.num_modes)
            val_minade_10 = min_xde_K(val_ade_losses, val_mode_probs, K=min(self.args.num_modes, 10))
            val_minade_5 = min_xde_K(val_ade_losses, val_mode_probs, K=5)
            val_minade_1 = min_xde_K(val_ade_losses, val_mode_probs, K=1)
            val_minfde_c = min_xde_K(val_fde_losses, val_mode_probs, K=self.args.num_modes)
            val_minfde_1 = min_xde_K(val_fde_losses, val_mode_probs, K=1)

            # Log val metrics
            # self.writer.add_scalar("metrics/Val minADE_{}".format(self.args.num_modes), val_minade_c[0], epoch)
            self.writer.add_scalar("metrics step/Val minADE_{}".format(10), val_minade_10[0], step)
            self.writer.add_scalar("metrics step/Val minADE_{}".format(5), val_minade_5[0], step)
            self.writer.add_scalar("metrics step/Val minADE_{}".format(1), val_minade_1[0], step)
            # self.writer.add_scalar("metrics/Val minFDE_{}".format(self.args.num_modes), val_minfde_c[0], epoch)
            self.writer.add_scalar("metrics step/Val minFDE_{}".format(1), val_minfde_1[0], step)
            if self.args.evaluate_causal:
                self.writer.add_scalar("metrics step/Val consistency", np.array(val_consistency).mean(), step)
                self.writer.add_scalar("metrics step/Val HNC", val_HNC, step)
                self.writer.add_scalar("metrics step/Val ARS", val_ARS, step)

                print("minADE c:", val_minade_c[0], "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                      "minFDE c:", val_minfde_c[0], "minFDE_1:", val_minfde_1[0], "Consistency:",
                      round(np.array(val_consistency).mean(), 2), "HNC:", val_HNC, "ARS:", round(val_ARS, 2))
            else:
                print("minADE c:", val_minade_c[0], "minADE_10", val_minade_10[0], "minADE_5", val_minade_5[0],
                      "minFDE c:", val_minfde_c[0], "minFDE_1:", val_minfde_1[0])
            self.autobot_model.train()
            self.save_model(minade_k=val_minade_c[0], minfde_k=val_minfde_c[0])

    def autobotjoint_train(self):
        steps = 0
        for epoch in range(0, self.args.num_epochs):
            print("Epoch:", epoch)
            epoch_marg_ade_losses = []
            epoch_marg_fde_losses = []
            epoch_marg_mode_probs = []
            epoch_scene_ade_losses = []
            epoch_scene_fde_losses = []
            epoch_mode_probs = []
            for i, data in enumerate(self.train_loader):
                ego_in, ego_out, agents_in, agents_out, map_lanes, agent_types = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, map_lanes, agent_types)

                nll_loss, kl_loss, post_entropy, adefde_loss = \
                    nll_loss_multimodes_joint(pred_obs, ego_out, agents_out, mode_probs,
                                              entropy_weight=self.args.entropy_weight,
                                              kl_weight=self.args.kl_weight,
                                              use_FDEADE_aux_loss=self.args.use_FDEADE_aux_loss,
                                              agent_types=agent_types,
                                              predict_yaw=self.predict_yaw)

                self.optimiser.zero_grad()
                (nll_loss + adefde_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.autobot_model.parameters(), self.args.grad_clip_norm)
                self.optimiser.step()

                self.writer.add_scalar("Loss/nll", nll_loss.item(), steps)
                self.writer.add_scalar("Loss/adefde", adefde_loss.item(), steps)
                self.writer.add_scalar("Loss/kl", kl_loss.item(), steps)

                with torch.no_grad():
                    ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                    epoch_marg_ade_losses.append(ade_losses.reshape(-1, self.args.num_modes))
                    epoch_marg_fde_losses.append(fde_losses.reshape(-1, self.args.num_modes))
                    epoch_marg_mode_probs.append(
                        mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                            -1, self.args.num_modes))

                    scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out)
                    epoch_scene_ade_losses.append(scene_ade_losses)
                    epoch_scene_fde_losses.append(scene_fde_losses)
                    epoch_mode_probs.append(mode_probs.detach().cpu().numpy())

                if i % 10 == 0:
                    print(i, "/", len(self.train_loader.dataset)//self.args.batch_size,
                          "NLL loss", round(nll_loss.item(), 2), "KL loss", round(kl_loss.item(), 2),
                          "Prior Entropy", round(torch.mean(D.Categorical(mode_probs).entropy()).item(), 2),
                          "Post Entropy", round(post_entropy, 2), "ADE+FDE loss", round(adefde_loss.item(), 2))

                steps += 1

            epoch_marg_ade_losses = np.concatenate(epoch_marg_ade_losses)
            epoch_marg_fde_losses = np.concatenate(epoch_marg_fde_losses)
            epoch_marg_mode_probs = np.concatenate(epoch_marg_mode_probs)
            epoch_scene_ade_losses = np.concatenate(epoch_scene_ade_losses)
            epoch_scene_fde_losses = np.concatenate(epoch_scene_fde_losses)
            mode_probs = np.concatenate(epoch_mode_probs)
            train_minade_c = min_xde_K(epoch_marg_ade_losses, epoch_marg_mode_probs, K=self.args.num_modes)
            train_minfde_c = min_xde_K(epoch_marg_fde_losses, epoch_marg_mode_probs, K=self.args.num_modes)
            train_sminade_c = min_xde_K(epoch_scene_ade_losses, mode_probs, K=self.args.num_modes)
            train_sminfde_c = min_xde_K(epoch_scene_fde_losses, mode_probs, K=self.args.num_modes)
            print("Train Marg. minADE c:", train_minade_c[0], "Train Marg. minFDE c:", train_minfde_c[0], "\n",
                  "Train Scene minADE c", train_sminade_c[0], "Train Scene minFDE c", train_sminfde_c[0])

            # Log train metrics
            self.writer.add_scalar("metrics/Train Marg. minADE {}".format(self.args.num_modes), train_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Train Marg. minFDE {}".format(self.args.num_modes), train_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Train Scene minADE {}".format(self.args.num_modes), train_sminade_c[0], epoch)
            self.writer.add_scalar("metrics/Train Scene minFDE {}".format(self.args.num_modes), train_sminfde_c[0], epoch)

            self.optimiser_scheduler.step()
            self.autobotjoint_evaluate(epoch)
            self.save_model(epoch)
            print("Best Scene minADE c", self.smallest_minade_k, "Best Scene minFDE c", self.smallest_minfde_k)

    def autobotjoint_evaluate(self, epoch):
        self.autobot_model.eval()
        with torch.no_grad():
            val_marg_ade_losses = []
            val_marg_fde_losses = []
            val_marg_mode_probs = []
            val_scene_ade_losses = []
            val_scene_fde_losses = []
            val_mode_probs = []
            for i, data in enumerate(self.val_loader):
                ego_in, ego_out, agents_in, agents_out, context_img, agent_types = self._data_to_device(data)
                pred_obs, mode_probs = self.autobot_model(ego_in, agents_in, context_img, agent_types)

                # Marginal metrics
                ade_losses, fde_losses = self._compute_marginal_errors(pred_obs, ego_out, agents_out, agents_in)
                val_marg_ade_losses.append(ade_losses.reshape(-1, self.args.num_modes))
                val_marg_fde_losses.append(fde_losses.reshape(-1, self.args.num_modes))
                val_marg_mode_probs.append(
                    mode_probs.unsqueeze(1).repeat(1, self.num_other_agents + 1, 1).detach().cpu().numpy().reshape(
                        -1, self.args.num_modes))

                # Joint metrics
                scene_ade_losses, scene_fde_losses = self._compute_joint_errors(pred_obs, ego_out, agents_out)
                val_scene_ade_losses.append(scene_ade_losses)
                val_scene_fde_losses.append(scene_fde_losses)
                val_mode_probs.append(mode_probs.detach().cpu().numpy())

            val_marg_ade_losses = np.concatenate(val_marg_ade_losses)
            val_marg_fde_losses = np.concatenate(val_marg_fde_losses)
            val_marg_mode_probs = np.concatenate(val_marg_mode_probs)

            val_scene_ade_losses = np.concatenate(val_scene_ade_losses)
            val_scene_fde_losses = np.concatenate(val_scene_fde_losses)
            val_mode_probs = np.concatenate(val_mode_probs)

            val_minade_c = min_xde_K(val_marg_ade_losses, val_marg_mode_probs, K=self.args.num_modes)
            val_minfde_c = min_xde_K(val_marg_fde_losses, val_marg_mode_probs, K=self.args.num_modes)
            val_sminade_c = min_xde_K(val_scene_ade_losses, val_mode_probs, K=self.args.num_modes)
            val_sminfde_c = min_xde_K(val_scene_fde_losses, val_mode_probs, K=self.args.num_modes)

            # Log train metrics
            self.writer.add_scalar("metrics/Val Marg. minADE {}".format(self.args.num_modes), val_minade_c[0], epoch)
            self.writer.add_scalar("metrics/Val Marg. minFDE {}".format(self.args.num_modes), val_minfde_c[0], epoch)
            self.writer.add_scalar("metrics/Val Scene minADE {}".format(self.args.num_modes), val_sminade_c[0], epoch)
            self.writer.add_scalar("metrics/Val Scene minFDE {}".format(self.args.num_modes), val_sminfde_c[0], epoch)

            print("Marg. minADE c:", val_minade_c[0], "Marg. minFDE c:", val_minfde_c[0])
            print("Scene minADE c:", val_sminade_c[0], "Scene minFDE c:", val_sminfde_c[0])

            self.autobot_model.train()
            self.save_model(minade_k=val_sminade_c[0], minfde_k=val_sminfde_c[0])

    def load_model(self):
        weights = torch.load(self.args.weight_path)
        self.autobot_model.load_state_dict(weights["AutoBot"], strict=False)

    def load_optimiser(self):
        weights = torch.load(self.args.weight_path)

        if self.args.reg_type in ["contrastive", "ranking"] and "contrastive_projector.0.weight" not in weights["AutoBot"].keys():
            for param_group in self.optimiser.param_groups:
                param_group['initial_lr'] = weights["optimiser"]['param_groups'][0]['initial_lr']
                param_group['lr'] = weights["optimiser"]['param_groups'][0]['lr']

            new_ids = [i for i, p in enumerate(self.autobot_model.named_parameters()) if not "contrastive_projector" in p[0]]
            assert len(new_ids) == len(weights["optimiser"]['state'])
            new_state_dict = {new_ids[i]: weights["optimiser"]['state'][i] for i in range(len(new_ids))}
            self.optimiser.state_dict()['state'] = new_state_dict
        else:
            self.optimiser.load_state_dict(weights["optimiser"])

    def save_model(self, epoch=None, minade_k=None, minfde_k=None):
        if epoch is None:
            if minade_k < self.smallest_minade_k:
                self.smallest_minade_k = minade_k
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_ade.pth"),
                )

            if minfde_k < self.smallest_minfde_k:
                self.smallest_minfde_k = minfde_k
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "best_models_fde.pth"),
                )

        else:
            if epoch % self.args.save_every == 0 and epoch > 0:
                torch.save(
                    {
                        "AutoBot": self.autobot_model.state_dict(),
                        "optimiser": self.optimiser.state_dict(),
                    },
                    os.path.join(self.results_dirname, "models_%d.pth" % epoch),
                )

    def train(self):
        if "Ego" in self.args.model_type:
            self.autobotego_train()
        elif "Joint" in self.args.model_type:
            self.autobotjoint_train()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    args, results_dirname = get_train_args()
    wandb.init(
        # set the wandb project where this run will be logged
        project="AutoBots",
        name="Contrastive" if args.reg_type=="contrastive" else "Ranking",
        # track hyperparameters and run metadata
        config={
            "W": args.contrastive_weight if args.reg_type=="contrastive" else args.ranking_weight,
            "rank_consecutive": args.do_consecutive if args.reg_type=="ranking" else None,
            "lr": args.learning_rate,
        }
    )
    trainer = Trainer(args, results_dirname)
    trainer.train()
