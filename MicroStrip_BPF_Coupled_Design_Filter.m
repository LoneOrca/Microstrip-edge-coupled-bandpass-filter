%% code for plotting MicroStrip_BPF_Coupled_Design_Filter
file = 'MicroStrip_BPF_Coupled_Design_Filter.s2p';
g = read(rfdata.data,file);
figure
plot(g,'s11','s21', 'db');