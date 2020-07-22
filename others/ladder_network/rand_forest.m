clc;
clear;

a=load('CellData_test');
test_label = a.y_test;
b=load('test-500');

acc = zeros(4,1);
Pred_double = str2num(cell2mat(Pred));
for i = 0:3
    idx = find(test_label == i);
    test_label_tmp = test_label(idx);
    pred_tmp = Pred_double(idx);
    acc(i) = mean(pred_tmp==test_label_tmp);
end



