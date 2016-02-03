function [train_best_acc,test_best_acc] = NPSVMLLP(data,npsvmPara)

%   [train_best_acc,test_best_acc] = NPSVMLLP(data,npsvmPara)
%   input:
%   data -- data.A:              postive instances of the training data.
%           data.B:              negative instances of the training data.
%           data.A_bag:          the bag's label of each positve instance.
%           data.B_bag:          the bag's label of each negative instance.
%           data.bagnum:         the total number of bags in the training data.
%           data.train_bag_prop: the label proportions of each bag in the training data.
%           data.testX:          the test data.
%           data.testY:          the labels of the test data.
%           npsvmPara:           the parameters about NPSVM. More details is available in the "Read Me for NPSVM.txt"  
%
%   output:
%         train_best_acc:        the accuracy of the training data.
%         test_best_acc:         the accuracy of the test data.
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%

data.X = [data.A;data.B]; data.Y = [ones(size(data.A,1),1);  -ones(size(data.B,1),1)];
N_random = 30;
%N_random = 200;
result = cell(N_random,1);
obj = Inf*ones(N_random*100,1);
Accuracy = zeros(N_random*100,1);

for i=1:N_random
    % simple metho no par selection.
    data_num = size(data.A,1) + size(data.B,1);
    init_y = ones(data_num,1);
    r = randperm(data_num);
    init_y(r(1:floor(data_num/2))) = -1;
    data.init_y = init_y;
    [result{i},Accuracy(i)] = NPSVMLLP_solve(data,npsvmPara);
    obj(i) = result{i}.obj;
end

[~,id]  = min(obj);
train_best_acc = Accuracy(id);
[~, ~,test_best_acc] = NPSVM_Test(data.testX,data.testY, result{id});

end







