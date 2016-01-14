function  [predicted_label, decision_values,accuracy] = NPSVM_Test(X,Y, model)

data_num = size(X,1);
[model0, model1] = splitmodel(model);

[predicted_label, accuracy1, decision_values1] = svmpredict(Y,X,model0);
[predicted_label, accuracy2, decision_values2] = svmpredict(Y, X,model1);
%    decision_values = decision_values1;
for i=1:data_num
    if((abs(decision_values1(i)))<(abs(decision_values2(i))))
        predicted_label(i) =1;
    else
        predicted_label(i) =-1;
    end
    decision_values = abs(decision_values2) - abs(decision_values1);
end

accuracy =  length(find(Y==predicted_label))/length(predicted_label);

end



function [model0, model1] = splitmodel(model)

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