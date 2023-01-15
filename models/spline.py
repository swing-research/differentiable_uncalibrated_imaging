'''
This code is an adaptation of
https://github.com/idealab-isu/NURBSDiff/blob/master/NURBSDiff/nurbs_eval.py
which was committed on November 3 2021
'''

import torch
import numpy as np
from torch import nn

class NURBS2D(torch.nn.Module):

    def __init__(self, control_pts, weights, u_spline_space, v_spline_space, deg_x, deg_y):
        super(NURBS2D, self).__init__()

        if type(control_pts) is np.ndarray:
            control_pts = torch.from_numpy(control_pts.astype(np.float32))

        self.control_pts = nn.Parameter(control_pts.cuda(), requires_grad=True)
        self.weights = nn.Parameter(weights.cuda(), requires_grad=True)

        self.u_spline_space = nn.Parameter(torch.from_numpy(u_spline_space.astype(np.float32)).cuda(), requires_grad=True)
        self.v_spline_space = torch.from_numpy(v_spline_space.astype(np.float32)).cuda()
        
        self.deg_x = deg_x
        self.deg_y = deg_y
        
        self._knot_u = self._create_knots(deg_x, self.control_pts.shape[1], False)
        self._knot_v = self._create_knots(deg_y, self.control_pts.shape[2], False)

        self._dimension = 3

        self.vspan_uv_forward, self.Nv_uv_forward = self._create_nonchanging_basis(
            self._knot_v, self.v_spline_space, self.deg_y)
        self.Nv_uv_forward = self.Nv_uv_forward.permute(1,0,2).unsqueeze(1).unsqueeze(-1).unsqueeze(-3)

        self.u_eval = None
        self.v_eval = None

        U_c = torch.cumsum(torch.where(self._knot_u<0.0, self._knot_u*0+1e-4, self._knot_u), dim=1)
        self.U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    def _create_knots(self, deg, n_ctrl_pts, nonuniform):
        zeros_1 = torch.zeros(deg+1).cuda()
        zeros_0 = torch.zeros(deg).cuda()
        knot = torch.ones(n_ctrl_pts - deg).cuda()
        if nonuniform:
            knot = torch.linspace(0+1e-5, 1-1e-5, n_ctrl_pts - deg).cuda()
        return torch.cat((zeros_1, knot, zeros_0)).unsqueeze(0)

    def _create_nonchanging_basis(self, knot_v, v, deg):
        # For when evaluation points do not change
        V_c = torch.cumsum(torch.where(knot_v<0.0, knot_v*0+1e-4, knot_v), dim=1)
        V = (V_c - V_c[:,0].unsqueeze(-1)) / (V_c[:,-1].unsqueeze(-1) - V_c[:,0].unsqueeze(-1))

        v = v.unsqueeze(0)
        vspan_uv = torch.stack([torch.min(torch.where((v - V[s,deg:-deg].unsqueeze(1))>1e-8, v - V[s,deg:-deg].unsqueeze(1), (v - V[s,deg:-deg].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+deg for s in range(V.size(0))])

        Ni = [v*0 for i in range(deg+1)]
        Ni[0] = v*0 + 1
        for k in range(1,deg+1):
            saved = (v)*0.0
            for r in range(k):
                VList1 = torch.stack([V[s,vspan_uv[s,:] + r + 1] for s in range(V.size(0))])
                VList2 = torch.stack([V[s,vspan_uv[s,:] + 1 - k + r] for s in range(V.size(0))])
                temp = Ni[r]/((VList1 - v) + (v - VList2))
                temp = torch.where(((VList1 - v) + (v - VList2))==0.0, v*0+1e-4, temp)
                Ni[r] = saved + (VList1 - v)*temp
                saved = (v - VList2)*temp
            Ni[k] = saved

        return vspan_uv, torch.stack(Ni)

    def eval(self, u_eval, v_eval):
        ctrl_pts =  torch.cat((self.control_pts * self.weights, self.weights), -1)

        U = self.U

        if self.u_eval is None or self.u_eval.shape != u_eval.shape or (self.u_eval != u_eval).all():
            self.u_eval = u_eval
            self.uspan_uv_eval, self.Nu_uv_eval = self._create_nonchanging_basis(
                self._knot_u, torch.from_numpy(u_eval.astype(np.float32)).cuda(), self.deg_x)
            self.Nu_uv_eval = self.Nu_uv_eval.permute(1,0,2).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        
        uspan_uv = self.uspan_uv_eval
        Nu_uv = self.Nu_uv_eval

        if self.v_eval is None or self.v_eval.shape != v_eval.shape or (self.v_eval != v_eval).all():
            self.v_eval = v_eval
            self.vspan_uv_eval, self.Nv_uv_eval = self._create_nonchanging_basis(
                self._knot_v, torch.from_numpy(v_eval.astype(np.float32)).cuda(), self.deg_y)
            self.Nv_uv_eval = self.Nv_uv_eval.permute(1,0,2).unsqueeze(1).unsqueeze(-1).unsqueeze(-3)
        
        vspan_uv = self.vspan_uv_eval
        Nv_uv = self.Nv_uv_eval

        pts = torch.stack([torch.stack([torch.stack([ctrl_pts[s,(uspan_uv[s,:]-self.deg_x+l),:,:][:,(vspan_uv[s,:]-self.deg_y+r),:] \
            for r in range(self.deg_y+1)]) for l in range(self.deg_x+1)]) for s in range(U.size(0))])

        surfaces = torch.sum((Nu_uv*pts)*Nv_uv, (1,2))

        surfaces = surfaces[:,:,:,:self._dimension]/surfaces[:,:,:,self._dimension].unsqueeze(-1)
        return surfaces


    def forward(self):
        
        ctrl_pts =  torch.cat((self.control_pts * self.weights, self.weights), -1)

        U = self.U

        if torch.isnan(U).any():
            print(U_c)
            print(knot_u)

        u = self.u_spline_space
        u = torch.sort(u)[0]

        u = u.unsqueeze(0)
        uspan_uv = torch.stack(
            [torch.min(
                torch.where(
                    (u - U[s,self.deg_x:-self.deg_x].unsqueeze(1))>1e-8, u - U[s,self.deg_x:-self.deg_x].unsqueeze(1), (u - U[s,self.deg_x:-self.deg_x].unsqueeze(1))*0.0 + 1
                    ),
                0,keepdim=False)[1]+self.deg_x for s in range(U.size(0))
            ]
            )
        uspan_uv = torch.where(uspan_uv == len(u[0]), len(u[0]) - 1, uspan_uv)

        u = u.squeeze(0)
        Ni = [u*0 for i in range(self.deg_x+1)]
        Ni[0] = u*0 + 1
        UList_dict = {}

        for i in range(1-self.deg_x, self.deg_x+1):
            if i not in UList_dict:
                UList_dict[i] = torch.stack([U[s,uspan_uv[s,:] + i] for s in range(U.size(0))])

        for k in range(1,self.deg_x+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = UList_dict[r + 1]
                UList2 = UList_dict[1 - k + r]
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(2).unsqueeze(-1).unsqueeze(-1)

        vspan_uv = self.vspan_uv_forward
        Nv_uv = self.Nv_uv_forward

        pts = torch.stack(
            [torch.stack(
                [torch.stack(
                    [ctrl_pts[s,(uspan_uv[s,:]-self.deg_x+l),:,:][:,(vspan_uv[s,:]-self.deg_y+r),:] \
            for r in range(self.deg_y+1)]) for l in range(self.deg_x+1)]) for s in range(U.size(0))])

        surfaces = torch.sum((Nu_uv*pts)*Nv_uv, (1,2))

        surfaces = surfaces[:,:,:,:self._dimension]/surfaces[:,:,:,self._dimension].unsqueeze(-1)
        return surfaces


class NURBS3D(torch.nn.Module):

    def __init__(self, control_pts, weights, u_spline_space, v_spline_space, w_spline_space, deg_x, deg_y, deg_z):
        super(NURBS3D, self).__init__()

        if type(control_pts) is np.ndarray:
            control_pts = torch.from_numpy(control_pts.astype(np.float32))

        self.control_pts = nn.Parameter(control_pts.cuda(), requires_grad=True)
        self.weights = nn.Parameter(weights.cuda(), requires_grad=True)

        self.u_spline_space = nn.Parameter(torch.from_numpy(u_spline_space.astype(np.float32)).cuda(), requires_grad=True)
        self.v_spline_space = torch.from_numpy(v_spline_space.astype(np.float32)).cuda()
        self.w_spline_space = torch.from_numpy(w_spline_space.astype(np.float32)).cuda()
        
        self.deg_x = deg_x
        self.deg_y = deg_y
        self.deg_z = deg_z
        
        self._knot_u = self._create_knots(deg_x, self.control_pts.shape[1], False)
        self._knot_v = self._create_knots(deg_y, self.control_pts.shape[2], False)
        self._knot_w = self._create_knots(deg_z, self.control_pts.shape[3], False)

        self._dimension = 4

        self.vspan_uv_forward, self.Nv_uv_forward = self._create_nonchanging_basis(
            self._knot_v, self.v_spline_space, self.deg_y)
        self.Nv_uv_forward = self.Nv_uv_forward.permute(1,0,2).unsqueeze(1).unsqueeze(3).unsqueeze(-1).unsqueeze(-3).unsqueeze(-1)

        self.wspan_uv_forward, self.Nw_uv_forward = self._create_nonchanging_basis(
            self._knot_w, self.w_spline_space, self.deg_z)
        self.Nw_uv_forward = self.Nw_uv_forward.permute(1,0,2).unsqueeze(1).unsqueeze(1).unsqueeze(-2).unsqueeze(-2).unsqueeze(-1)

        self.u_eval = None
        self.v_eval = None
        self.w_eval = None

        U_c = torch.cumsum(torch.where(self._knot_u<0.0, self._knot_u*0+1e-4, self._knot_u), dim=1)
        self.U = (U_c - U_c[:,0].unsqueeze(-1)) / (U_c[:,-1].unsqueeze(-1) - U_c[:,0].unsqueeze(-1))

    def _create_knots(self, deg, n_ctrl_pts, nonuniform):
        zeros_1 = torch.zeros(deg+1).cuda()
        zeros_0 = torch.zeros(deg).cuda()
        knot = torch.ones(n_ctrl_pts - deg).cuda()
        if nonuniform:
            knot = torch.linspace(0+1e-5, 1-1e-5, n_ctrl_pts - deg).cuda()
        return torch.cat((zeros_1, knot, zeros_0)).unsqueeze(0)

    def _create_nonchanging_basis(self, knot_v, v, deg):
        # For when evaluation points do not change
        V_c = torch.cumsum(torch.where(knot_v<0.0, knot_v*0+1e-4, knot_v), dim=1)
        V = (V_c - V_c[:,0].unsqueeze(-1)) / (V_c[:,-1].unsqueeze(-1) - V_c[:,0].unsqueeze(-1))

        v = v.unsqueeze(0)
        vspan_uv = torch.stack([torch.min(torch.where((v - V[s,deg:-deg].unsqueeze(1))>1e-8, v - V[s,deg:-deg].unsqueeze(1), (v - V[s,deg:-deg].unsqueeze(1))*0.0 + 1),0,keepdim=False)[1]+deg for s in range(V.size(0))])

        Ni = [v*0 for i in range(deg+1)]
        Ni[0] = v*0 + 1
        for k in range(1,deg+1):
            saved = (v)*0.0
            for r in range(k):
                VList1 = torch.stack([V[s,vspan_uv[s,:] + r + 1] for s in range(V.size(0))])
                VList2 = torch.stack([V[s,vspan_uv[s,:] + 1 - k + r] for s in range(V.size(0))])
                temp = Ni[r]/((VList1 - v) + (v - VList2))
                temp = torch.where(((VList1 - v) + (v - VList2))==0.0, v*0+1e-4, temp)
                Ni[r] = saved + (VList1 - v)*temp
                saved = (v - VList2)*temp
            Ni[k] = saved

        return vspan_uv, torch.stack(Ni)

    def eval(self, u_eval, v_eval, w_eval):
        ctrl_pts =  torch.cat((self.control_pts, self.weights), -1)

        U = self.U

        if self.u_eval is None or self.u_eval.shape != u_eval.shape or (self.u_eval != u_eval).all():
            self.u_eval = u_eval
            self.uspan_uv_eval, self.Nu_uv_eval = self._create_nonchanging_basis(
                self._knot_u, torch.from_numpy(u_eval.astype(np.float32)).cuda(), self.deg_x)
            self.Nu_uv_eval = self.Nu_uv_eval.permute(1,0,2).unsqueeze(2).unsqueeze(2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        uspan_uv = self.uspan_uv_eval
        Nu_uv = self.Nu_uv_eval

        if self.v_eval is None or self.v_eval.shape != v_eval.shape or (self.v_eval != v_eval).all():
            self.v_eval = v_eval
            self.vspan_uv_eval, self.Nv_uv_eval = self._create_nonchanging_basis(
                self._knot_v, torch.from_numpy(v_eval.astype(np.float32)).cuda(), self.deg_y)
            self.Nv_uv_eval = self.Nv_uv_eval.permute(1,0,2).unsqueeze(1).unsqueeze(3).unsqueeze(-1).unsqueeze(-3).unsqueeze(-1)
        
        vspan_uv = self.vspan_uv_eval
        Nv_uv = self.Nv_uv_eval

        if self.w_eval is None or self.w_eval.shape != w_eval.shape or (self.w_eval != w_eval).all():
            self.w_eval = w_eval
            self.wspan_uv_eval, self.Nw_uv_eval = self._create_nonchanging_basis(
                self._knot_w, torch.from_numpy(w_eval.astype(np.float32)).cuda(), self.deg_z)
            self.Nw_uv_eval = self.Nw_uv_eval.permute(1,0,2).unsqueeze(1).unsqueeze(1).unsqueeze(-2).unsqueeze(-2).unsqueeze(-1)

        wspan_uv = self.wspan_uv_eval
        Nw_uv = self.Nw_uv_eval

        running_val = torch.zeros([1,1,1,1,len(u_eval), ctrl_pts.shape[2], ctrl_pts.shape[3], self._dimension + 1]).cuda()
        for i in range(self.deg_x + 1):
            for j in range(self.deg_y + 1):
                for k in range(self.deg_z + 1):
                    p_ijk = ctrl_pts[0,(uspan_uv[0]-self.deg_x+i)][:,(vspan_uv[0]-self.deg_y+j)][:,:,(wspan_uv[0]-self.deg_z+k)].unsqueeze(0)
                    Nu_uv_i = Nu_uv[:,i]
                    Nv_uv_j = Nv_uv[:,:,j]
                    Nw_uv_k = Nw_uv[:,:,:,k]
                    running_val += p_ijk * Nu_uv_i * Nv_uv_j * Nw_uv_k

        surfaces = running_val.squeeze().unsqueeze(0)
        surfaces = surfaces[:,:,:,:,:self._dimension]
        return surfaces

    def forward(self):        
        ctrl_pts =  torch.cat((self.control_pts, self.weights), -1)
        U = self.U

        if torch.isnan(U).any():
            print(U_c)
            print(knot_u)

        u = self.u_spline_space
        u = torch.sort(u)[0]
        # v = self.v_spline_space

        u = u.unsqueeze(0)
        pts_minus_knots = u - U[0,self.deg_x:-self.deg_x].unsqueeze(1)
        uspan_uv = torch.min(
            torch.where(pts_minus_knots>1e-8, pts_minus_knots, pts_minus_knots*0.0 + 1),
                0,keepdim=False)[1]+self.deg_x
        uspan_uv = uspan_uv.unsqueeze(0)
        uspan_uv = torch.where(uspan_uv == len(u[0]), len(u[0]) - 1, uspan_uv)

        u = u.squeeze(0)
        Ni = [u*0 for i in range(self.deg_x+1)]
        Ni[0] = u*0 + 1
        UList_dict = {}

        for i in range(1-self.deg_x, self.deg_x+1):
            if i not in UList_dict:
                UList_dict[i] = (U[0,uspan_uv[0] + i]).unsqueeze(0)

        for k in range(1,self.deg_x+1):
            saved = (u)*0.0
            for r in range(k):
                UList1 = UList_dict[r + 1]
                UList2 = UList_dict[1 - k + r]
                temp = Ni[r]/((UList1 - u) + (u - UList2))
                temp = torch.where(((UList1 - u) + (u - UList2))==0.0, u*0+1e-4, temp)
                Ni[r] = saved + (UList1 - u)*temp
                saved = (u - UList2)*temp
            Ni[k] = saved

        Nu_uv = torch.stack(Ni).permute(1,0,2).unsqueeze(2).unsqueeze(2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        vspan_uv = self.vspan_uv_forward
        Nv_uv = self.Nv_uv_forward

        wspan_uv = self.wspan_uv_forward
        Nw_uv = self.Nw_uv_forward

        running_val = torch.zeros([1,1,1,1,ctrl_pts.shape[1], ctrl_pts.shape[2], ctrl_pts.shape[3], self._dimension + 1]).cuda()
        for i in range(self.deg_x + 1):
            for j in range(self.deg_y + 1):
                for k in range(self.deg_z + 1):
                    p_ijk = ctrl_pts[0,(uspan_uv[0]-self.deg_x+i)][:,(vspan_uv[0]-self.deg_y+j)][:,:,(wspan_uv[0]-self.deg_z+k)].unsqueeze(0)
                    Nu_uv_i = Nu_uv[:,i]
                    Nv_uv_j = Nv_uv[:,:,j]
                    Nw_uv_k = Nw_uv[:,:,:,k]
                    running_val += p_ijk * Nu_uv_i * Nv_uv_j * Nw_uv_k

        surfaces = running_val.squeeze().unsqueeze(0)
        surfaces = surfaces[:,:,:,:,:self._dimension]
        return surfaces
