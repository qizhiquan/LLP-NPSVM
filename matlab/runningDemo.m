%  A demo for LLP-NPSVM
%
%   input:
%   data -- data.A:              postive instances of the training data.
%           data.B:              negative instances of the training data.
%           data.A_bag:          the bag's label of each positve instance.
%           data.B_bag:          the bag's label of each negative instance.
%           data.bagnum:         the total number of bags in the training data.
%           data.train_bag_prop: the label proportions of each bag in the training data.
%           data.testX:          the test data.
%           data.testY:          the labels of the test data.
%
%   output:
%         train_best_acc:        the accuracy of the training data.
%         test_best_acc:         the accuracy of the test data.
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%



close all; clear;
load toydata.mat

npsvmPara = NPSVM_Parameters();
npsvmPara.KernelType = 'linear';
npsvmPara.C = 0.01;
%npsvmPara.KernelParas = 0.05;
npsvmPara.Epsilon = 0.1;

tic;
   [train_best_acc,test_best_acc] = NPSVMLLP(data,npsvmPara)
toc;






         






