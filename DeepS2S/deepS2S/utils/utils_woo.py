#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def cassou_regime_analysis(mjo_inputs, loop_probabilities, loop_targets, frac = 0.8, qall_90 = False):

    # conditional probabilities.
    t_in, t_out = mjo_inputs.shape[2],loop_probabilities.shape[2]
    sampled_cassous_ls = []
    for mod in range(loop_probabilities.shape[0]): 
        sub_indx = np.random.choice(np.arange(loop_probabilities.shape[0]), 
                                    size = int(frac*loop_probabilities.shape[0]), replace = False)
        sub_mjo = mjo_inputs[sub_indx,...]
        sub_lp = loop_probabilities[sub_indx,...]
        sub_targets = loop_targets[sub_indx,...]

        cassou_conditional_count = np.zeros((4,8,t_in +t_out-1))
        cassou_unconditional_count = np.zeros((4,t_in +t_out-1))
        for i in range(sub_lp.shape[0]):
            for j in range(sub_lp.shape[1]):
                for k in range(sub_lp.shape[2]):
                    for t in range(mjo_inputs.shape[2]):
                        if not np.isnan(mjo_inputs[i,j,k]):
                            if not qall_90:
                                d_t = np.abs((t-5)) + (k+1)
                                cassou_conditional_count[sub_targets[i,j,k], int(mjo_inputs[i,j,t])-1, d_t-1] += 1
                                cassou_unconditional_count[sub_targets[i,j,k], d_t-1] += 1
                            else:
                                if sub_lp[i,j,k,sub_targets[i,j,k]] > qall_90:
                                    d_t = np.abs((t-5)) + (k+1)
                                    cassou_conditional_count[sub_targets[i,j,k], int(mjo_inputs[i,j,t])-1, d_t-1] += 1
                                    cassou_unconditional_count[sub_targets[i,j,k], d_t-1] += 1


        cassou_conditional_total = np.sum(cassou_conditional_count, axis = 0)
        cassou_unconditional_total = np.sum(cassou_unconditional_count, axis = 0)
        cassou_conditional_pc = cassou_conditional_count/np.repeat(cassou_conditional_total[None,:,:],4,axis=0)
        cassou_unconditional_pc = cassou_unconditional_count/np.repeat(cassou_unconditional_total[None,:],4,axis=0)
             
        cassou_probability_change = 100*(cassou_conditional_pc - np.repeat(cassou_unconditional_pc[:,None,:],8,axis=1))

        sampled_cassous_ls.append(cassou_probability_change[None,...])

    sampled_cassous = np.concatenate(sampled_cassous_ls)

    probability_cassou_ls = []
    probability_cassou_array = np.zeros((4,8,t_in +t_out-1))
    for mjo_phase in range(9):
        for shift_value in range(t_in +t_out-1):
            for label in range(4):
                probability_change = sampled_cassous[:,label,mjo_phase,shift_value]

                mean = np.mean(probability_change)
                pc_975 = np.nanpercentile(probability_change ,97.5)
                pc_025 = np.nanpercentile(probability_change, 2.5)

                if np.sign(pc_025)==np.sign(pc_975):
                    significance = 1
                else:
                    significance = 0
    
                probability_cassou_ls.append(pd.DataFrame(data={
                    "mjo phase": [mjo_phase],
                    "shift": [shift_value],
                    "label": [label],
                    "mean": [mean],
                    "significance": [significance]}))
                
                probability_cassou_array[label,mjo_phase,shift_value] = mean

    probability_cassou = pd.concat(probability_cassou_ls)


    return probability_cassou, probability_cassou_array, sampled_cassous

def nae_regimes_analysis(nae_inputs, 
                         regimes, 
                         loop_probabilities, 
                         loop_targets, 
                         frac = 0.8, 
                         qall_90 = False):
    # conditional probabilities.
    frac = 0.8
    t_in, t_out = nae_inputs.shape[2],loop_probabilities.shape[2]
    sampled_nae_ls = []
    for mod in range(loop_probabilities.shape[0]): 
        sub_indx = np.random.choice(np.arange(loop_probabilities.shape[0]), 
                                    size = int(frac*loop_probabilities.shape[0]), replace = False)
        sub_nae = nae_inputs[sub_indx,...]
        sub_lp = loop_probabilities[sub_indx,...]
        sub_targets = loop_targets[sub_indx,...]

        nae_conditional_count = np.zeros((len(regimes),len(regimes),t_in +t_out-1))
        nae_unconditional_count = np.zeros((len(regimes),t_in +t_out-1))
        for i in range(sub_lp.shape[0]):
            for j in range(sub_lp.shape[1]):
                for k in range(sub_lp.shape[2]):
                    for t in range(nae_inputs.shape[2]):
                        if not np.isnan(sub_nae[i,j,k]):
                            if not qall_90:
                                d_t = np.abs((t-5)) + (k+1)
                                nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]), d_t-1] += 1
                                nae_unconditional_count[sub_targets[i,j,k], d_t-1] += 1
                            else:
                                if sub_lp[i,j,k,sub_targets[i,j,k]] > qall_90:
                                    d_t = np.abs((t-5)) + (k+1)
                                    nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]), d_t-1] += 1
                                    nae_unconditional_count[sub_targets[i,j,k], d_t-1] += 1


        nae_conditional_total = np.sum(nae_conditional_count, axis = 0)
        nae_unconditional_total = np.sum(nae_unconditional_count, axis = 0)
        nae_conditional_pc = nae_conditional_count/np.repeat(nae_conditional_total[None,:,:],len(regimes),axis=0)
        nae_unconditional_pc = nae_unconditional_count/np.repeat(nae_unconditional_total[None,:],len(regimes),axis=0)
                
        nae_probability_change = 100 *(nae_conditional_pc - np.repeat(nae_unconditional_pc[:,None,:],len(regimes),axis=1))

        sampled_nae_ls.append(nae_probability_change[None,...])

    sampled_naes = np.concatenate(sampled_nae_ls)

    probability_nae_ls = []
    probability_nae_array = np.zeros((len(regimes),len(regimes),t_in +t_out-1))
    for r in range(len(regimes)):
        for shift_value in range(t_in +t_out-1):
            for label in range(len(regimes)):
                probability_change = sampled_naes[:,label,r,shift_value]

                mean = np.mean(probability_change)
                pc_975 = np.nanpercentile(probability_change ,97.5)
                pc_025 = np.nanpercentile(probability_change, 2.5)

                if np.sign(pc_025)==np.sign(pc_975):
                    significance = 1
                else:
                    significance = 0

                probability_nae_ls.append(pd.DataFrame(data={
                    "NAE regime": [r+1],
                    "Lag (weeks)": [shift_value+1],
                    "predicted NAE regime": [label+1],
                    "mean": [mean],
                    "significance": [significance]}))
                
                probability_nae_array[label,r,shift_value] = mean

    probability_nae = pd.concat(probability_nae_ls)

    return probability_nae, probability_nae_array, sampled_naes

def nae_regimes_analysis_timesteps(nae_inputs, 
                         regimes, 
                         loop_probabilities, 
                         loop_targets, 
                         frac = 0.8, 
                         qall_90 = False):
    # conditional probabilities.
    frac = 0.8
    t_in, t_out = nae_inputs.shape[2],loop_probabilities.shape[2]
    sampled_nae_ls = []
    for mod in range(loop_probabilities.shape[0]): 
        sub_indx = np.random.choice(np.arange(loop_probabilities.shape[0]), 
                                    size = int(frac*loop_probabilities.shape[0]), replace = False)
        sub_nae = nae_inputs[sub_indx,...]
        sub_lp = loop_probabilities[sub_indx,...]
        sub_targets = loop_targets[sub_indx,...]

        nae_conditional_count = np.zeros((len(regimes),len(regimes),sub_lp.shape[2], t_in +t_out-1))
        nae_unconditional_count = np.zeros((len(regimes),sub_lp.shape[2], t_in +t_out-1))
        nae_conditional_count[:] = np.nan
        nae_unconditional_count[:] = np.nan
        for i in range(sub_lp.shape[0]):
            for j in range(sub_lp.shape[1]):
                for k in range(sub_lp.shape[2]):
                    for t in range(nae_inputs.shape[2]):
                        if not np.isnan(sub_nae[i,j,k]):
                            if not qall_90:
                                d_t = np.abs((t-5)) + (k+1)
                                if np.isnan(nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]),k, d_t-1]):
                                    nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]),k, d_t-1] = 1
                                else:
                                    nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]),k, d_t-1] += 1
                                if np.isnan(nae_unconditional_count[sub_targets[i,j,k],k, d_t-1]):
                                    nae_unconditional_count[sub_targets[i,j,k],k, d_t-1] = 1
                                else:
                                    nae_unconditional_count[sub_targets[i,j,k],k, d_t-1] += 1
                            else:
                                if sub_lp[i,j,k,sub_targets[i,j,k]] > qall_90:
                                    d_t = np.abs((t-5)) + (k+1)
                                    if np.isnan(nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]),k, d_t-1]):
                                        nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]),k, d_t-1] = 1
                                    else:
                                        nae_conditional_count[sub_targets[i,j,k], int(sub_nae[i,j,t]),k, d_t-1] += 1
                                    if np.isnan(nae_unconditional_count[sub_targets[i,j,k],k, d_t-1]):
                                        nae_unconditional_count[sub_targets[i,j,k],k, d_t-1] = 1
                                    else:
                                        nae_unconditional_count[sub_targets[i,j,k],k, d_t-1] += 1

        nae_conditional_total = np.nansum(nae_conditional_count, axis = 0)
        nae_unconditional_total = np.nansum(nae_unconditional_count, axis = 0)
        nae_conditional_pc = nae_conditional_count/np.repeat(nae_conditional_total[None,:,:],len(regimes),axis=0)
        nae_unconditional_pc = nae_unconditional_count/np.repeat(nae_unconditional_total[None,:],len(regimes),axis=0)
                
        nae_probability_change = 100 *(nae_conditional_pc - np.repeat(nae_unconditional_pc[:,None,:],len(regimes),axis=1))

        sampled_nae_ls.append(nae_probability_change[None,...])

    sampled_naes = np.concatenate(sampled_nae_ls)

    probability_nae_ls = []
    probability_nae_array = np.zeros((len(regimes),len(regimes), nae_inputs.shape[2], t_in +t_out-1))
    for r in range(len(regimes)):
        for k in range(loop_probabilities.shape[2]):
            for shift_value in range(t_in +t_out-1):
                for label in range(len(regimes)):
                    probability_change = sampled_naes[:,label,r,k,shift_value]

                    mean = np.nanmean(probability_change)
                    pc_975 = np.nanpercentile(probability_change ,97.5)
                    pc_025 = np.nanpercentile(probability_change, 2.5)

                    if np.sign(pc_025)==np.sign(pc_975):
                        significance = 1
                    else:
                        significance = 0

                    probability_nae_ls.append(pd.DataFrame(data={
                        "NAE regime": [r+1],
                        "timestep": [k+1],
                        "Lag (weeks)": [shift_value+1],
                        "predicted NAE regime": [label+1],
                        "mean": [mean],
                        "significance": [significance]}))
                    
                    probability_nae_array[label,r,k,shift_value] = mean

    probability_nae = pd.concat(probability_nae_ls) 

    return probability_nae, probability_nae_array, sampled_naes

