%% code for plotting MCFIL_BPF_Coupled_Tuned_v2
file = 'MCFIL_BPF_Coupled_Tuned_v2.s2p';
g = read(rfdata.data,file);
figure
plot(g,'s11', 's21', 'db');