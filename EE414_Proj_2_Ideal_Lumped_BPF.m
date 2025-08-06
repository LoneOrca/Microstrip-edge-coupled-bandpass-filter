%----------------------------------------------------------------------
% Srinivas N - 620094 
%----------------------------------------------------------------------
% EE 414-514
%  BPF Design Project-2
%----------------------------------------------------------------------

clearvars
clc
close all

%----------------------------------------------------------------------
% Units 
%----------------------------------------------------------------------

G = 10^9;
Meg = 10^6;
k = 10^+3;
c = 10^-2;
m = 10^-3;
u = 10^-6;
n = 10^-9;
%----------------------------------------------------------------------
%----------------------------------------------------------------------
IFigure = 0;
NF = 32;
dfreq = 1;
df = 1*Meg;
j = 1*j;
theta_P_f0 = 90;
%----------------------------------------------------------------------
% 
%---------------------------------------------------------------------
fls = 7.25*G;
fhs = 9*G;

fh = 8.4*G; % where?
fl = 7.6*G; % where?

f0 = sqrt((fh*fl));

delta = ((fh - fl)/f0);
BWf = fh - fl;
Print_Real_Unit('f0',f0,'Hz')
Print_Real_Unit('Bwf',BWf,'Hz')
Print_Real('delta',delta)

%----------------------------------------------------------------------
ILmin = 0.5; % dB
S21_dB = -ILmin;
S21min = 10^((S21_dB)/20);
Print_Real('S21_Min(IL)',S21min,'W/W'); 

S11_max = sqrt(1-abs(S21min)^2); % W/W
Print_Real('S11_max (IL)',S11_max,'W/W'); 
S11_dB = 20*log10(S11_max); % IL

Print_Real('S11_dB (IL)',S11_dB,'W/W'); 

RLmin = 22;

S11_dB_RL = -RLmin;

S11_max_RL = 10^(((S11_dB_RL)/20));

S21min_RL = sqrt(1-abs(S11_max_RL)^2);
Print_Real('S21_min (RL)',S21min_RL,'W/W'); 
S21_dB_RL = 20*log10(S21min_RL); % IL
Print_Real('S21_min (RL)',S21_dB_RL,'dB'); 

Ap_dB = 0.0200; % Round down
Print_Real('Ap (RIPPLE)',Ap_dB,'dB'); 

%----------------------------------------------------------------------

wp = 2*pi*f0;
Ap = 10^(Ap_dB/10); % W/W
Xp = Ap - 1;
epsilon = sqrt(Xp);
%----------------------------------------------------------------------
% FLS
%----------------------------------------------------------------------

ALS = 40;
ALS_W = 10^(ALS/10);
XLS = ALS_W - 1;
Omega_LS = (1/delta)*((fls/f0)-(f0/fls));
NLS = acosh(sqrt(XLS)/epsilon ) / acosh(abs(Omega_LS));
Print_Real('NLS',NLS)
%----------------------------------------------------------------------
% FHS
%----------------------------------------------------------------------
AHS = 50;
AHS_W = 10^(AHS/10);
XHS = AHS_W - 1;
Omega_HS = (1/delta)*((fhs/f0)-(f0/fhs));
NHS = acosh(sqrt(XHS)/epsilon ) / acosh(abs(Omega_HS));
Print_Real('NHS',NHS)

NP = 7; % Round up

Z0 = 50;
Y0 = 1/Z0;
w0 = 2*pi*f0;
Print_Real_Unit('w0',w0,'Grad/s')
g0 = 1.0000;
g1 = 0.8859;
g2 = 1.4217;
g3 = 1.8300;
g4 = 1.6372;
g5 = 1.8300;
g6 = 1.4217;
g7 = 0.8859;
g8 = 1.0000;
%----------------------------------------------------------------------
% Branch #1 (Shunt)
%----------------------------------------------------------------------
C1 = (1/(w0*delta))*g1*Y0;
L1 = 1/(w0^2*C1);
Print_Break
Print_Real_Unit('C1',C1,'F')
Print_Real_Unit('L1',L1,'H')
%----------------------------------------------------------------------
% Branch #2 (Series)
%----------------------------------------------------------------------
L2 = (1/(w0*delta))*g2*Z0;
C2 = 1/(w0^2*L2);
Print_Real_Unit('L2',L2,'H')
Print_Real_Unit('C2',C2,'F')
%----------------------------------------------------------------------
% Branch #3 (Shunt)
%----------------------------------------------------------------------
C3 = (1/(w0*delta))*g3*Y0;
L3 = 1/(w0^2*C3);
Print_Real_Unit('C3',C3,'F')
Print_Real_Unit('L3',L3,'H')
%----------------------------------------------------------------------
% Branch #4 (Series)
%----------------------------------------------------------------------
L4 = (1/(w0*delta))*g4*Z0;
C4 = 1/(w0^2*L4);
Print_Real_Unit('L4',L4,'H')
Print_Real_Unit('C4',C4,'F')
%----------------------------------------------------------------------
% Branch #5 (Shunt)
%----------------------------------------------------------------------
C5 = (1/(w0*delta))*g5*Y0;
L5 = 1/(w0^2*C5);
Print_Real_Unit('C5',C5,'F')
Print_Real_Unit('L5',L5,'H')
%----------------------------------------------------------------------
% Branch #6 (Series)
%----------------------------------------------------------------------
L6 = (1/(w0*delta))*g6* Z0;
C6 = 1/(w0^2*L6);
Print_Real_Unit('L6',L6,'H')
Print_Real_Unit('C6',C6,'F')
%----------------------------------------------------------------------
% Branch #7 (Shunt)
%----------------------------------------------------------------------
C7 = (1/(w0*delta))*g7*Y0;
L7 = 1/(w0^2*C7);
Print_Real_Unit('C7',C7,'F')
Print_Real_Unit('L7',L7,'H')

f_min = 7*G;
f_max = 9*G;
freq = f_min : df : f_max;
freq = sort(freq);
freq = freq';
%I_f0_Amp = freq_Amp == f0;

N_Freq = length(freq);
S_Filter = zeros(N_Freq,2,2);
for kk = 1 : N_Freq
fk = freq(kk);
T0 = eye(2); % ?
T1 = EE414_ABCD_Shunt_C1(C1, fk);
T2 = EE414_ABCD_Shunt_L1(L1, fk);
T3 = EE414_ABCD_Series_L2(L2,fk);
T4 = EE414_ABCD_Series_C2(C2,fk);
T5 = EE414_ABCD_Shunt_C3(C3, fk);
T6 = EE414_ABCD_Shunt_L3(L3, fk);
T7 = EE414_ABCD_Series_L4(L4,fk);
T8 = EE414_ABCD_Series_C4(C4,fk);
T9 = EE414_ABCD_Shunt_C5(C5, fk);
T10 = EE414_ABCD_Shunt_L5(L5, fk);
T11 = EE414_ABCD_Series_L6(L6,fk);
T12 = EE414_ABCD_Series_C6(C6,fk);
T13 = EE414_ABCD_Shunt_C7(C7, fk);
T14 = EE414_ABCD_Shunt_L7(L7, fk);
T = T0*T1*T2*T3*T4*T5*T6*T7*T8*T9*T10*T11*T12*T13*T14;
S_Filter(kk, :, :) = ABCD_to_S(T, [Z0, Z0]); 
end

% %----------------------------------------------------------------------
% %
% %----------------------------------------------------------------------

S11_Filter = S_Filter(:,1,1);
S11_Filter_Mag = abs(S11_Filter);
S11_Filter_dB = 20*log10(S11_Filter_Mag);
S21_Filter = S_Filter(:,2,1);
S21_Filter_Mag = abs(S21_Filter);
S21_Filter_dB = 20*log10(S21_Filter_Mag);

% Section 01

j01 = sqrt(((1/2)*pi) * delta / (g0 * g1));
z0e_01 = Z0 * (1 + (j01) + (j01)^2);
z0o_01 = Z0 *(1 - (j01) + (j01)^2);

% Section 12

j12 = ((1/2)*pi) * delta / sqrt((g1 * g2));
z0e_12 = Z0 * (1 + (j12) + (j12)^2);
z0o_12 = Z0 *(1 - (j12) + (j12)^2);

% Section 23

j23 = ((1/2)*pi) * delta / sqrt((g2 * g3));
z0e_23 = Z0 * (1 + (j23) + (j23)^2);
z0o_23 = Z0 *(1 - (j23) + (j23)^2);

% Section 34


j34 = ((1/2)*pi) * delta / sqrt((g3 * g4));
z0e_34 = Z0 * (1 + (j34) + (j34)^2);
z0o_34 = Z0 *(1 - (j34) + (j34)^2);

% Section 45


j45 = ((1/2)*pi) * delta / sqrt((g4 * g5));
z0e_45 = Z0 * (1 + (j45) + (j45)^2);
z0o_45 = Z0 *(1 - (j45) + (j45)^2);

% Section 56


j56 = ((1/2)*pi) * delta / sqrt((g5 * g6));
z0e_56 = Z0 * (1 + (j56) + (j56)^2);
z0o_56 = Z0 *(1 - (j56) + (j56)^2);

% Section 67


j67 = ((1/2)*pi) * delta / sqrt((g6 * g7));
z0e_67 = Z0 * (1 + (j67) + (j67)^2);
z0o_67 = Z0 *(1 - (j67) + (j67)^2);

% Section 78

j78 = sqrt(((1/2)*pi) * delta / (g7 * g8));
z0e_78 = Z0 * (1 + (j78) + (j78)^2);
z0o_78 = Z0 *(1 - (j78) + (j78)^2);



Print_Real_Unit('j01',j01,'S/S')
Print_Real_Unit('z0e_01',z0e_01,'Ohms')
Print_Real_Unit('z0o_01',z0o_01,'Ohms')

Print_Real_Unit('j12',j12,'S/S')
Print_Real_Unit('z0e_12',z0e_12,'Ohms')
Print_Real_Unit('z0o_12',z0o_12,'Ohms')

Print_Real_Unit('j23',j23,'S/S')
Print_Real_Unit('z0e_23',z0e_23,'Ohms')
Print_Real_Unit('z0o_23',z0o_23,'Ohms')

Print_Real_Unit('j34',j34,'S/S')
Print_Real_Unit('z0e_34',z0e_34,'Ohms')
Print_Real_Unit('z0o_34',z0o_34,'Ohms')

Print_Real_Unit('j45',j45,'S/S')
Print_Real_Unit('z0e_45',z0e_45,'Ohms')
Print_Real_Unit('z0o_45',z0o_45,'Ohms')

Print_Real_Unit('j56',j56,'S/S')
Print_Real_Unit('z0e_56',z0e_56,'Ohms')
Print_Real_Unit('z0o_56',z0o_56,'Ohms')

Print_Real_Unit('j67',j67,'S/S')
Print_Real_Unit('z0e_67',z0e_67,'Ohms')
Print_Real_Unit('z0o_67',z0o_67,'Ohms')

Print_Real_Unit('j78',j78,'S/S')
Print_Real_Unit('z0e_78',z0e_78,'Ohms')
Print_Real_Unit('z0o_78',z0o_78,'Ohms')


%j01_y0 = j01/Y0
er = 6.45;
W = 1.77*m;
h = 1.27*m;

er_eff = (1/2)*(er +1)+((er-1)/(2*sqrt(1+12*(h/W))));
Print_Real('er_eff',er_eff)
delta_L = 0.412*h*((er_eff+0.3)/(er_eff - 0.258))*(((W/h)+0.264)/((W/h)+0.8));
Print_Real_Unit('delta_L',delta_L,'m')
%%
% %----------------------------------------------------------------------
% % Plots
% %----------------------------------------------------------------------
IFigure = IFigure + 1;
figure_max(IFigure)
plot(freq/G, S11_Filter_dB, 'g', 'linewidth', 6)
hold on
plot(freq/G, S21_Filter_dB, 'r', 'linewidth', 6);
plot(f0/G,interp1(freq/G,S21_Filter_dB,f0/G),'ro','linewidth',10);
plot(fl/G,interp1(freq/G,S21_Filter_dB,fl/G),'ro','linewidth',10);
plot(fh/G,interp1(freq/G,S21_Filter_dB,fh/G),'ro','linewidth',10);
plot(fl/G,interp1(freq/G,S11_Filter_dB,fl/G),'go','linewidth',10);
plot(fh/G,interp1(freq/G,S11_Filter_dB,fh/G),'go','linewidth',10);

hold off
grid on
grid minor 
xlabel('{\itf} (GHz) ')
ylabel('| {\itS}_{k1} | ( dB ) ', ...
'VerticalAlignment', 'bottom')
set(gca, 'FontName', 'times new roman', 'FontSize', NF)
legend('|{\itS}_{11}|','|{\itS}_{21}|')
axis([f_min/G, f_max/G, -40, 0])
set(gca, 'xtick', f_min/G : 0.1 : f_max/G);
set(gca, 'linewidth', 2.0)


IFigure = IFigure + 1;
figure_max(IFigure)
plot(freq/G, S21_Filter_dB, 'r', 'linewidth', 6)
hold on
plot(f0/G,interp1(freq/G,S21_Filter_dB,f0/G),'ro','linewidth',10);
plot(fl/G,interp1(freq/G,S21_Filter_dB,fl/G),'ro','linewidth',10);
plot(fh/G,interp1(freq/G,S21_Filter_dB,fh/G),'ro','linewidth',10);
hold off
grid on
grid minor 
xlabel('{\itf} (GHz) ')
ylabel('| {\itS}_{21} | ( dB ) ', ...
'VerticalAlignment', 'bottom')
set(gca, 'FontName', 'times new roman', 'FontSize', NF)
legend('|{\itS}_{21}|')
axis([7*G/G, 9*G/G, -0.03, 0])
set(gca, 'xtick', f_min/G : 0.1 : f_max/G);
set(gca, 'linewidth', 2.0)
%----------------------------------------------------------------------
% EE 414
%  BPF Design Example
%----------------------------------------------------------------------

clearvars
clc
close all

%----------------------------------------------------------------------
% Units 
%----------------------------------------------------------------------

G = 10^9;
Meg = 10^6;
k = 10^+3;
c = 10^-2;
m = 10^-3;
u = 10^-6;
n = 10^-9;
%----------------------------------------------------------------------
%----------------------------------------------------------------------
IFigure = 0;
NF = 32;
dfreq = 1;
df = 1*Meg;
j = 1*j;
theta_P_f0 = 90;
%----------------------------------------------------------------------
% 
%---------------------------------------------------------------------
fls = 7.25*G;
fhs = 9*G;

fh = 8.4*G; % where?
fl = 7.6*G; % where?

f0 = sqrt((fh*fl));

delta = ((fh - fl)/f0);
BWf = fh - fl;
Print_Real_Unit('f0',f0,'Hz')
Print_Real_Unit('Bwf',BWf,'Hz')
Print_Real('delta',delta)

%----------------------------------------------------------------------
ILmin = 0.5; % dB
S21_dB = -ILmin;
S21min = 10^((S21_dB)/20);
Print_Real('S21_Min(IL)',S21min,'W/W'); 

S11_max = sqrt(1-abs(S21min)^2); % W/W
Print_Real('S11_max (IL)',S11_max,'W/W'); 
S11_dB = 20*log10(S11_max); % IL

Print_Real('S11_dB (IL)',S11_dB,'W/W'); 

RLmin = 22;

S11_dB_RL = -RLmin;

S11_max_RL = 10^(((S11_dB_RL)/20));

S21min_RL = sqrt(1-abs(S11_max_RL)^2);
Print_Real('S21_min (RL)',S21min_RL,'W/W'); 
S21_dB_RL = 20*log10(S21min_RL); % IL
Print_Real('S21_min (RL)',S21_dB_RL,'dB'); 

Ap_dB = 0.0200; % Round down
Print_Real('Ap (RIPPLE)',Ap_dB,'dB'); 

%----------------------------------------------------------------------

wp = 2*pi*f0;
Ap = 10^(Ap_dB/10); % W/W
Xp = Ap - 1;
epsilon = sqrt(Xp);
%----------------------------------------------------------------------
% FLS
%----------------------------------------------------------------------

ALS = 40;
ALS_W = 10^(ALS/10);
XLS = ALS_W - 1;
Omega_LS = (1/delta)*((fls/f0)-(f0/fls));
NLS = acosh(sqrt(XLS)/epsilon ) / acosh(abs(Omega_LS));
Print_Real('NLS',NLS)
%----------------------------------------------------------------------
% FHS
%----------------------------------------------------------------------
AHS = 50;
AHS_W = 10^(AHS/10);
XHS = AHS_W - 1;
Omega_HS = (1/delta)*((fhs/f0)-(f0/fhs));
NHS = acosh(sqrt(XHS)/epsilon ) / acosh(abs(Omega_HS));
Print_Real('NHS',NHS)

NP = 7; % Round up

Z0 = 50;
Y0 = 1/Z0;
w0 = 2*pi*f0;
Print_Real_Unit('w0',w0,'Grad/s')
g0 = 1.0000;
g1 = 0.8859;
g2 = 1.4217;
g3 = 1.8300;
g4 = 1.6372;
g5 = 1.8300;
g6 = 1.4217;
g7 = 0.8859;
g8 = 1.0000;
%----------------------------------------------------------------------
% Branch #1 (Shunt)
%----------------------------------------------------------------------
C1 = (1/(w0*delta))*g1*Y0;
L1 = 1/(w0^2*C1);
Print_Break
Print_Real_Unit('C1',C1,'F')
Print_Real_Unit('L1',L1,'H')
%----------------------------------------------------------------------
% Branch #2 (Series)
%----------------------------------------------------------------------
L2 = (1/(w0*delta))*g2*Z0;
C2 = 1/(w0^2*L2);
Print_Real_Unit('L2',L2,'H')
Print_Real_Unit('C2',C2,'F')
%----------------------------------------------------------------------
% Branch #3 (Shunt)
%----------------------------------------------------------------------
C3 = (1/(w0*delta))*g3*Y0;
L3 = 1/(w0^2*C3);
Print_Real_Unit('C3',C3,'F')
Print_Real_Unit('L3',L3,'H')
%----------------------------------------------------------------------
% Branch #4 (Series)
%----------------------------------------------------------------------
L4 = (1/(w0*delta))*g4*Z0;
C4 = 1/(w0^2*L4);
Print_Real_Unit('L4',L4,'H')
Print_Real_Unit('C4',C4,'F')
%----------------------------------------------------------------------
% Branch #5 (Shunt)
%----------------------------------------------------------------------
C5 = (1/(w0*delta))*g5*Y0;
L5 = 1/(w0^2*C5);
Print_Real_Unit('C5',C5,'F')
Print_Real_Unit('L5',L5,'H')
%----------------------------------------------------------------------
% Branch #6 (Series)
%----------------------------------------------------------------------
L6 = (1/(w0*delta))*g6* Z0;
C6 = 1/(w0^2*L6);
Print_Real_Unit('L6',L6,'H')
Print_Real_Unit('C6',C6,'F')
%----------------------------------------------------------------------
% Branch #7 (Shunt)
%----------------------------------------------------------------------
C7 = (1/(w0*delta))*g7*Y0;
L7 = 1/(w0^2*C7);
Print_Real_Unit('C7',C7,'F')
Print_Real_Unit('L7',L7,'H')

f_min = 7*G;
f_max = 9*G;
freq = f_min : df : f_max;
freq = sort(freq);
freq = freq';
%I_f0_Amp = freq_Amp == f0;

N_Freq = length(freq);
S_Filter = zeros(N_Freq,2,2);
for kk = 1 : N_Freq
fk = freq(kk);
T0 = eye(2); % ?
T1 = EE414_ABCD_Shunt_C1(C1, fk);
T2 = EE414_ABCD_Shunt_L1(L1, fk);
T3 = EE414_ABCD_Series_L2(L2,fk);
T4 = EE414_ABCD_Series_C2(C2,fk);
T5 = EE414_ABCD_Shunt_C3(C3, fk);
T6 = EE414_ABCD_Shunt_L3(L3, fk);
T7 = EE414_ABCD_Series_L4(L4,fk);
T8 = EE414_ABCD_Series_C4(C4,fk);
T9 = EE414_ABCD_Shunt_C5(C5, fk);
T10 = EE414_ABCD_Shunt_L5(L5, fk);
T11 = EE414_ABCD_Series_L6(L6,fk);
T12 = EE414_ABCD_Series_C6(C6,fk);
T13 = EE414_ABCD_Shunt_C7(C7, fk);
T14 = EE414_ABCD_Shunt_L7(L7, fk);
T = T0*T1*T2*T3*T4*T5*T6*T7*T8*T9*T10*T11*T12*T13*T14;
S_Filter(kk, :, :) = ABCD_to_S(T, [Z0, Z0]); 
end

% %----------------------------------------------------------------------
% %
% %----------------------------------------------------------------------

S11_Filter = S_Filter(:,1,1);
S11_Filter_Mag = abs(S11_Filter);
S11_Filter_dB = 20*log10(S11_Filter_Mag);
S21_Filter = S_Filter(:,2,1);
S21_Filter_Mag = abs(S21_Filter);
S21_Filter_dB = 20*log10(S21_Filter_Mag);

% Section 01

j01 = sqrt(((1/2)*pi) * delta / (g0 * g1));
z0e_01 = Z0 * (1 + (j01) + (j01)^2);
z0o_01 = Z0 *(1 - (j01) + (j01)^2);

% Section 12

j12 = ((1/2)*pi) * delta / sqrt((g1 * g2));
z0e_12 = Z0 * (1 + (j12) + (j12)^2);
z0o_12 = Z0 *(1 - (j12) + (j12)^2);

% Section 23

j23 = ((1/2)*pi) * delta / sqrt((g2 * g3));
z0e_23 = Z0 * (1 + (j23) + (j23)^2);
z0o_23 = Z0 *(1 - (j23) + (j23)^2);

% Section 34


j34 = ((1/2)*pi) * delta / sqrt((g3 * g4));
z0e_34 = Z0 * (1 + (j34) + (j34)^2);
z0o_34 = Z0 *(1 - (j34) + (j34)^2);

% Section 45


j45 = ((1/2)*pi) * delta / sqrt((g4 * g5));
z0e_45 = Z0 * (1 + (j45) + (j45)^2);
z0o_45 = Z0 *(1 - (j45) + (j45)^2);

% Section 56


j56 = ((1/2)*pi) * delta / sqrt((g5 * g6));
z0e_56 = Z0 * (1 + (j56) + (j56)^2);
z0o_56 = Z0 *(1 - (j56) + (j56)^2);

% Section 67


j67 = ((1/2)*pi) * delta / sqrt((g6 * g7));
z0e_67 = Z0 * (1 + (j67) + (j67)^2);
z0o_67 = Z0 *(1 - (j67) + (j67)^2);

% Section 78

j78 = sqrt(((1/2)*pi) * delta / (g7 * g8));
z0e_78 = Z0 * (1 + (j78) + (j78)^2);
z0o_78 = Z0 *(1 - (j78) + (j78)^2);



Print_Real_Unit('j01',j01,'S/S')
Print_Real_Unit('z0e_01',z0e_01,'Ohms')
Print_Real_Unit('z0o_01',z0o_01,'Ohms')

Print_Real_Unit('j12',j12,'S/S')
Print_Real_Unit('z0e_12',z0e_12,'Ohms')
Print_Real_Unit('z0o_12',z0o_12,'Ohms')

Print_Real_Unit('j23',j23,'S/S')
Print_Real_Unit('z0e_23',z0e_23,'Ohms')
Print_Real_Unit('z0o_23',z0o_23,'Ohms')

Print_Real_Unit('j34',j34,'S/S')
Print_Real_Unit('z0e_34',z0e_34,'Ohms')
Print_Real_Unit('z0o_34',z0o_34,'Ohms')

Print_Real_Unit('j45',j45,'S/S')
Print_Real_Unit('z0e_45',z0e_45,'Ohms')
Print_Real_Unit('z0o_45',z0o_45,'Ohms')

Print_Real_Unit('j56',j56,'S/S')
Print_Real_Unit('z0e_56',z0e_56,'Ohms')
Print_Real_Unit('z0o_56',z0o_56,'Ohms')

Print_Real_Unit('j67',j67,'S/S')
Print_Real_Unit('z0e_67',z0e_67,'Ohms')
Print_Real_Unit('z0o_67',z0o_67,'Ohms')

Print_Real_Unit('j78',j78,'S/S')
Print_Real_Unit('z0e_78',z0e_78,'Ohms')
Print_Real_Unit('z0o_78',z0o_78,'Ohms')


%j01_y0 = j01/Y0
er = 10.7;
W = 0.5541*m;
h = 0.635*m;

er_eff = (1/2)*(er +1)+((er-1)/(2*sqrt(1+12*(h/W))));
Print_Real('er_eff',er_eff)
delta_L = 0.412*h*((er_eff+0.3)/(er_eff - 0.258))*(((W/h)+0.264)/((W/h)+0.8));
Print_Real_Unit('delta_L',delta_L,'m')
%%
% %----------------------------------------------------------------------
% % Plots
% %----------------------------------------------------------------------
IFigure = IFigure + 1;
figure_max(IFigure)
plot(freq/G, S11_Filter_dB, 'g', 'linewidth', 6)
hold on
plot(freq/G, S21_Filter_dB, 'r', 'linewidth', 6);
plot(f0/G,interp1(freq/G,S21_Filter_dB,f0/G),'ro','linewidth',10);
plot(fl/G,interp1(freq/G,S21_Filter_dB,fl/G),'ro','linewidth',10);
plot(fh/G,interp1(freq/G,S21_Filter_dB,fh/G),'ro','linewidth',10);
plot(fl/G,interp1(freq/G,S11_Filter_dB,fl/G),'go','linewidth',10);
plot(fh/G,interp1(freq/G,S11_Filter_dB,fh/G),'go','linewidth',10);

hold off
grid on
grid minor 
xlabel('{\itf} (GHz) ')
ylabel('| {\itS}_{k1} | ( dB ) ', ...
'VerticalAlignment', 'bottom')
set(gca, 'FontName', 'times new roman', 'FontSize', NF)
legend('|{\itS}_{11}|','|{\itS}_{21}|')
axis([f_min/G, f_max/G, -40, 0])
set(gca, 'xtick', f_min/G : 0.1 : f_max/G);
set(gca, 'linewidth', 2.0)


IFigure = IFigure + 1;
figure_max(IFigure)
plot(freq/G, S21_Filter_dB, 'r', 'linewidth', 6)
hold on
plot(f0/G,interp1(freq/G,S21_Filter_dB,f0/G),'ro','linewidth',10);
plot(fl/G,interp1(freq/G,S21_Filter_dB,fl/G),'ro','linewidth',10);
plot(fh/G,interp1(freq/G,S21_Filter_dB,fh/G),'ro','linewidth',10);
hold off
grid on
grid minor 
xlabel('{\itf} (GHz) ')
ylabel('| {\itS}_{21} | ( dB ) ', ...
'VerticalAlignment', 'bottom')
set(gca, 'FontName', 'times new roman', 'FontSize', NF)
legend('|{\itS}_{21}|')
axis([7*G/G, 9*G/G, -40, 0])
set(gca, 'xtick', f_min/G : 0.1 : f_max/G);
set(gca, 'linewidth', 2.0)

%----------------------------------------------------
%% code for plotting BPF_Coupled_Line_Ideal_TRL
file = 'BPF_coupled_Ideal_TRL.s2p';
g = read(rfdata.data,file);
figure
plot(g,'s11','S21', 'db');
