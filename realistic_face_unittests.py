#!/usr/bin/env python3
# -*- coding: utf-8 -*-


########################################################################
# PROGRAMMER: Pierre-Antoine Ksinant                                   #
# DATE CREATED: 03/12/2018                                             #
# REVISED DATE: -                                                      #
# PURPOSE: This file contains unit tests for various functions used in #
#          the associated Jupyter Notebook, core of the project.       #
########################################################################


##################
# Needed imports #
##################

import numpy as np
import torch
from unittest.mock import MagicMock, patch


#################################
# Default tests success message #
#################################

def _print_success_message():
    print("Test passed!")


###################
# Class for tests #
###################

class AssertTest(object):
    
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])

    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message
        

##########################
# Function discriminator #
##########################

def test_discriminator(Discriminator):
    
    # Define parameters:
    batch_size = 50
    conv_dim=10
    D = Discriminator(conv_dim)

    # Create random image input:
    x = torch.from_numpy(np.random.randint(1, size=(batch_size, 3, 32, 32))*2 -1).float()
    
    # Check if GPU mode is available:
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        x.cuda()

    # Perform test:
    output = D(x)
    assert_test = AssertTest({
        'Conv Dim': conv_dim,
        'Batch Size': batch_size,
        'Input': x})

    # Check results:
    correct_output_size = (batch_size, 1)
    assert_condition = output.size() == correct_output_size
    assert_message = 'Wrong output size: Expected type {}, but got type {}...'.format(correct_output_size, output.size())
    assert_test.test(assert_condition, assert_message)

    # Test success message:
    _print_success_message()

    
######################
# Function generator #
######################

def test_generator(Generator):
    
    # Define parameters:
    batch_size = 50
    z_size = 25
    conv_dim=10
    G = Generator(z_size, conv_dim)

    # Create random input:
    z = np.random.uniform(-1, 1, size=(batch_size, z_size))
    z = torch.from_numpy(z).float()
    
    # Check if GPU mode is available:
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        z.cuda()

    # Perform test:
    output = G(z)
    assert_test = AssertTest({
        'Z size': z_size,
        'Conv Dim': conv_dim,
        'Batch Size': batch_size,
        'Input': z})

    # Check results:
    correct_output_size = (batch_size, 3, 32, 32)
    assert_condition = output.size() == correct_output_size
    assert_message = 'Wrong output size: Expected type {}, but got type {}...'.format(correct_output_size, output.size())
    assert_test.test(assert_condition, assert_message)

    # Test success message:
    _print_success_message()
