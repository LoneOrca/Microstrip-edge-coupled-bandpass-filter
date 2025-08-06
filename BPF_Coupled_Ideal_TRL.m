%% code for plotting BPF_Coupled_Line_Ideal_TRL
file = 'BPF_coupled_Ideal_TRL.s2p';
g = read(rfdata.data,file);
figure
plot(g,'s11','S21', 'db');

