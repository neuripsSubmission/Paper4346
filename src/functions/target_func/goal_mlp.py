import torch
import torch.nn as nn
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.geo import UP
import numpy as np
import quaternion
from src.utils.sim_utils import se3_to_mat


class GoalMLP(torch.nn.Module):
    def __init__(self):
        super(GoalMLP, self).__init__()
        self.f1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True))
        self.regression = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.dist_predict = torch.nn.Linear(256, 1)
        self.rot_predict = torch.nn.Linear(256, 1)
        self.switch_predict = torch.nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.5)
        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.dist_predict.weight.data)
        self.dist_predict.bias.data.zero_()
        nn.init.xavier_uniform_(self.rot_predict.weight.data)
        self.rot_predict.bias.data.zero_()
        nn.init.xavier_uniform_(self.switch_predict.weight.data)
        self.switch_predict.bias.data.zero_()

    def forward(self, x1, x2):
        x = self.dropout(self.regression(torch.mul(self.f1(x1), self.f1(x2)).squeeze()))
        dist_output = self.dist_predict(x)
        rot_output = self.rot_predict(x)
        switch_output = torch.sigmoid(self.switch_predict(x))
        return dist_output, rot_output, switch_output


class XRN(object):
    def __init__(self, opt):
        self.opt = opt
        self.dist_criterion = nn.SmoothL1Loss()
        self.rot_criterion = nn.SmoothL1Loss()
        self.switch_criterion = nn.BCELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = GoalMLP()
        self.model.to(self.device)
        self.learning_rate = 0.0001
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        self.finetune = False

    def set_lr(self, new_lr):
        self.learning_rate = new_lr
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train_start(self):
        self.model.train()

    def eval_start(self):
        self.model.eval()

    def eval_aux(
        self,
        rot_output,
        dist_output,
        true_phi,
        dist_score,
        start_poses,
        start_rots,
        goal_poses,
    ):
        goal_err = []
        for phi, rho, start_pos, start_rot, goal_pos in zip(
            rot_output, dist_output, start_poses, start_rots, goal_poses
        ):
            stateA = se3_to_mat(
                quaternion.from_float_array(start_rot),
                np.asarray(start_pos),
            )
            stateB = (
                stateA
                @ se3_to_mat(
                    quat_from_angle_axis(phi, UP),
                    np.asarray([0, 0, 0]),
                )
                @ se3_to_mat(
                    quaternion.from_float_array([1, 0, 0, 0]),
                    np.asarray([0, 0, -1 * rho]),
                )
            )
            final_pos = stateB[0:3, 3]
            goal_err.append(np.linalg.norm(goal_pos - final_pos))
        return np.mean(goal_err)

    def run_emb(
        self,
        batch_node1,
        batch_node2,
        true_phi,
        angle_encoding,
        dist_score,
        start_poses,
        start_rots,
        goal_poses,
        switch,
        *args
    ):
        # get model output
        dist_output, rot_output, switch_output = self.model(
            batch_node1.to(self.device), batch_node2.to(self.device)
        )

        # losses
        switch_loss = self.switch_criterion(
            switch_output.squeeze(1).float(), switch.to(self.device).float()
        )

        dist_output = dist_output.squeeze(1)
        dist_loss = self.dist_criterion(
            dist_output.float(),
            dist_score.to(self.device).float(),
        )

        rot_output = rot_output.squeeze(1)
        rot_loss = self.rot_criterion(
            rot_output.float(),
            true_phi.to(self.device),
        )
        losses = [dist_loss.item(), rot_loss.item(), switch_loss.item()]
        if self.finetune:
            loss = dist_loss + rot_loss
        else:
            loss = dist_loss + rot_loss + switch_loss

        # err, acc, switch ratio
        location_err = self.eval_aux(
            rot_output.detach().cpu(),
            dist_output.detach().cpu(),
            true_phi,
            dist_score,
            start_poses,
            start_rots,
            goal_poses,
        )
        switch_acc = (
            torch.eq(
                switch.detach().cpu(), np.round(switch_output.detach().cpu()).squeeze()
            ).sum()
            * 1.0
            / switch.size()[0]
        )

        phis = np.rad2deg(rot_output.cpu().detach().numpy())
        return (
            loss,
            losses,
            switch_acc,
            location_err,
            dist_output.cpu().detach().numpy(),
            phis,
        )

    def train_emb(self, *batch_data):
        self.optimizer.zero_grad()
        loss, losses, switch_acc, location_err, dist_output, phis = self.run_emb(
            *batch_data
        )
        loss.backward()
        self.optimizer.step()
        return (
            loss,
            losses,
            switch_acc,
            location_err,
            dist_output,
            phis,
        )
