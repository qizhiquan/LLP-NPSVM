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

[mm,id]  = min(obj);
train_best_acc = Accuracy(id);
[predicted_label, decision_values,test_best_acc] = NPSVM_Test(data.testX,data.testY, result{id});

end



function [Model,Accuracy] = NPSVMLLP_solve(data,npsvmPara)

%   [Model,Accuracy] = NPSVMLLP_solve(data,npsvmPara)
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
%         Model:                 the output model of the LLP-NPSVM.
%         Accuracy:              the accuracy of the training data. 
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%




a1 = data.init_y==1; b1 = data.init_y==-1; A2 = data.X(a1,:);B2 = data.X(b1,:);
iteration = 20; accuracy=zeros(iteration,1); compute_num=1; obj=[0 0]; Model=[];

for j=1:iteration
    X_temp= [A2;B2];
    label_temp = [ones(size(A2,1),1);  -ones(size(B2,1),1)];
    if j>=2 model_pre = model;end
    
    [model] = NPSVM_Train(X_temp, label_temp, npsvmPara);
    [predicted_label, decision_values,accur] = NPSVM_Test(data.X,data.Y, model);
    
    if(var(predicted_label)==0),
        Model=model; Accuracy=0;
        Model.obj=Inf;
        break;
    end;
    
    accuracy(j) = accur;
    
    [A2,B2] = grouping1(data,decision_values);
    
    if (j>=2&&compute_num<2)
        [flag] = stopguidelines(model_pre,model);
        if (flag==1||j == iteration);
            obj(1)  = computeobj(A2,B2,model);
            %           obj(1) = computeratio(data,model);
            compute_num  = compute_num+1;
            Model        = model;
            model        = swapmodel(model);
            [predicted_label, decision_values] = NPSVM_Test(data.X,data.Y,model);
            [A2,B2] = grouping1(data,decision_values);
            obj(2)  = computeobj(A2,B2,model);
            %obj(2) = computeratio(data,model);
            if  obj(1)<obj(2)
                Model  = Model;
                Model.obj(1)  = obj(1);
                [predicted_label, decision_values,Accuracy] = NPSVM_Test(data.X,data.Y, Model);
                %disp('model1');
                break;
            end
        end
    end
    if (compute_num==2)
        [flag] = stopguidelines(model_pre,model);
        
        if (flag==1||j == iteration)
            obj(2)  = computeobj(A2,B2,model);
            %obj(2) = computeratio(data,model);
            if(obj(1)<obj(2));
                Model  = Model;
                Model.obj(1)  = obj(1);
                %disp('model1');
            else
                Model = model;
                Model.obj(1)  = obj(2);
                %disp('model2');
            end
               [predicted_label, decision_values,Accuracy] = NPSVM_Test(data.X,data.Y, Model);
            break;
        end
    end
    
end

end


function [model0, model1] = splitmodel(model)

%   [model0, model1] = splitmodel(model)
%   input:
%         model:   the output model of the LLP-NPSVM.
%   output:
%         model0:  the model for the assumed positive class.
%         model1:  the model for the assumed negative class.
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%
%1
model0.Parameters = model.Parameters01;
model0.nr_class   = model.nr_class01;
model0.totalSV    = model.totalSV01;
model0.rho        = model.rho01;
model0.Label      = model.Label01;
model0.ProbA      = model.ProbA01;
model0.ProbB      = model.ProbB01;
model0.nSV        = model.nSV01;
model0.sv_coef    = model.sv_coef01;
model0.SVs        = model.SVs01;

%2
model1.Parameters = model.Parameters02;
model1.nr_class   = model.nr_class02;
model1.totalSV    = model.totalSV02;
model1.rho        = model.rho02;
model1.Label      = model.Label02;
model1.ProbA      = model.ProbA02;
model1.ProbB      = model.ProbB02;
model1.nSV        = model.nSV02;
model1.sv_coef    = model.sv_coef02;
model1.SVs        = model.SVs02;
end

function [A2,B2] = grouping1(data,decision_values)

%   [A2,B2] = grouping1(data,decision_values)
%   input:
% data -- data.A:              postive instances of the training data.
%         data.B:              negative instances of the training data.
%         data.A_bag:          the bag's label of each positve instance.
%         data.B_bag:          the bag's label of each negative instance.
%         data.bagnum:         the total number of bags in the training data.
%         data.train_bag_prop: the label proportions of each bag in the training data.
%         data.testX:          the test data.
%         data.testY:          the labels of the test data.
%         decision_values:     the predicted labels in the i-th iteration.
%   output:
%         A2:                  the assumed positive class in the i-th iteration.
%         B2:                  the assumed  negative class in the i-th iteration.
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%

bagnum = length(data.train_bag_prop);
[m1,n1] = size(data.A);
[m2,n2] = size(data.B);
a1=[]; b1=[];
for i=1:bagnum
    
    array = [find(data.A_bag==i)',find(data.B_bag==i)'+m1];
    part_values = decision_values(array);
    [DV,I] = sort(part_values,1,'descend');
    array_new = array(I);
    point = length(find(data.A_bag==i));
    a1 = [a1 array_new(1:point)];
    b1 = [b1 array_new(point+1:end)];
end

A2 = data.X(a1,:);
B2 = data.X(b1,:);
end


function model2 = swapmodel(model)

%   [model0, model1] = swapmodel(model)
%   input:
%         model:   the output model of the LLP-NPSVM.
%   output:
%         model2:  the model of swapping the assumed positive class and the  assumed negative class

%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%
%1
model2.Parameters01 = model.Parameters02;
model2.nr_class01   = model.nr_class02;
model2.totalSV01    = model.totalSV02;
model2.rho01        = model.rho02;
model2.Label01      = model.Label02;
model2.ProbA01      = model.ProbA02;
model2.ProbB01     = model.ProbB02;
model2.nSV01       = model.nSV02;
model2.sv_coef01   = model.sv_coef02;
model2.SVs01       = model.SVs02;

%2
model2.Parameters02 = model.Parameters01;
model2.nr_class02   = model.nr_class01;
model2.totalSV02    = model.totalSV01;
model2.rho02        = model.rho01;
model2.Label02      = model.Label01;
model2.ProbA02      = model.ProbA01;
model2.ProbB02     = model.ProbB01;
model2.nSV02       = model.nSV01;
model2.sv_coef02   = model.sv_coef01;
model2.SVs02       = model.SVs01;

end


function obj = computeobj(A,B,model)

%   obj = computeobj(A,B,model)
%   input:
%         A:       postive instances of the training data.
%         B:       negative instances of the training data.
%         model:   the output model of the LLP-NPSVM.
%   output:
%         obj:     the  average distance of all intances  to two   hyperplanes

%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%
 [model0, model1] = splitmodel(model);
 labelA = ones(size(A,1),1);
 labelB = ones(size(B,1),1);
 [predicted_label, accuracy1, decision_values1] = svmpredict(labelA,A,model0);
 [predicted_label, accuracy2, decision_values2] = svmpredict(labelB,B,model1);
 obj = mean(norm(decision_values1) + norm(decision_values2));
end

function bag_acc = computeratio(data,model)

%   bag_acc = computeratio(data,model)
%   input:
%data --  data.A:              postive instances of the training data.
%         data.B:              negative instances of the training data.
%         data.A_bag:          the bag's label of each positve instance.
%         data.B_bag:          the bag's label of each negative instance.
%         data.bagnum:         the total number of bags in the training data.
%         data.train_bag_prop: the label proportions of each bag in the training data.
%         data.testX:          the test data.
%         data.testY:          the labels of the test data.
%         model:               the output model of the LLP-NPSVM.
%   output:
%         bag_acc:             the average  accuracy about label proportions of bags in the training data.

%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%
X = data.X;
Y =data.Y;
data_num = size(X,1);
[model0, model1] = splitmodel(model);

[predicted_label, accuracy1, decision_values1] = svmpredict(Y,X,model0);
[predicted_label, accuracy2, decision_values2] = svmpredict(Y, X,model1);

for i=1:data_num
    if((abs(decision_values1(i)))<(abs(decision_values2(i))))
        predicted_label(i) =1;
    else
        predicted_label(i) =-1;
    end
    
end

AB_bag  = [data.A_bag;data.B_bag];
sum     = 0;
for i =1: data.bagnum
    predicted_label_temp=predicted_label(AB_bag ==i);
    ratio = data.train_bag_prop(i);
    sum = abs(sum+ length(find(predicted_label_temp==1))/length(predicted_label_temp)-ratio);
end
bag_acc =sum/data.bagnum;

end


function flag = stopguidelines(model_pre,model_new)

%   flag = stopguidelines(model_pre,model_new)
%   input:
%         model_pre:  the previous model during the i-th iteration.
%         model_new:  the current model during the i-th iteration.
%   output:
%         flag:       the index of stopping guidelines.
%
%   Author: Zhiquan Qi
%   Date: 2016.01.05
%
dif1 = norm(abs([model_pre.sv_coef01'*model_pre.SVs01 model_pre.rho01]-[model_new.sv_coef01'*model_new.SVs01 model_new.rho01]));
dif2 = norm(abs([model_pre.sv_coef02'*model_pre.SVs02 model_pre.rho02]-[model_new.sv_coef02'*model_new.SVs02 model_new.rho02]));

if dif1<0.015&&dif2<0.015
    flag=1;
else
    flag=0;
end

end



