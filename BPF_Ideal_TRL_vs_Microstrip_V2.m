%% code for plotting BPF Coupled Ideal TRL Vs Microstrip MCFIL V2 tuned
file = 'BPF_coupled_Ideal_TRL.s2p';
file2  = 'MCFIL_BPF_Coupled_Tuned_v2.s2p';
g = read(rfdata.data,file);
h = read(rfdata.data,file2);
figure
plot(g,'s11','s21','db');
hold all
plot(h,'s11','s21','db');

