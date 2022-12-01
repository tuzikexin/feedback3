import numpy as np
import pandas as pd
import random as rd

sub1 = pd.read_csv('./out/deberta_base.csv').sort_values(['text_id']).reset_index(drop=True)
sub2 = pd.read_csv('./out/deberta_v3_base.csv').sort_values(['text_id']).reset_index(drop=True)
sub3 = pd.read_csv('./out/deberta_v3_small.csv').sort_values(['text_id']).reset_index(drop=True)
sub4 = pd.read_csv('./out/lsg_base_4096.csv').sort_values(['text_id']).reset_index(drop=True)
sub5 = pd.read_csv('./out/lsg_electra_base.csv').sort_values(['text_id']).reset_index(drop=True)
sub6 = pd.read_csv('./out/deberta_large.csv').sort_values(['text_id']).reset_index(drop=True)
sub7 = pd.read_csv('./out/deberta_v3_large.csv').sort_values(['text_id']).reset_index(drop=True)
sub8 = pd.read_csv('./out/lsg_roberta_large_c1.csv').sort_values(['text_id']).reset_index(drop=True)
sub9 = pd.read_csv('./out/lsg_electra_large_c1.csv').sort_values(['text_id']).reset_index(drop=True)
sub10 = pd.read_csv('./out/deberta_v2_xlarge.csv').sort_values(['text_id']).reset_index(drop=True)
sub11 = pd.read_csv('./out/pseudo_deberta_v2_xlarge.csv').sort_values(['text_id']).reset_index(drop=True)
sub12 = pd.read_csv('./out/deberta_xlarge.csv').sort_values(['text_id']).reset_index(drop=True)
sub13 = pd.read_csv('./out/pseudo_deberta_xlarge.csv').sort_values(['text_id']).reset_index(drop=True)
sub14 = pd.read_csv('./out/ori_deberta_v3_base.csv').sort_values(['text_id']).reset_index(drop=True)
sub15 = pd.read_csv('./out/svr_v1.csv').sort_values(['text_id']).reset_index(drop=True)
sub16 = pd.read_csv('./out/svr_v2.csv').sort_values(['text_id']).reset_index(drop=True)
sub17 = pd.read_csv('./out/svr_v3.csv').sort_values(['text_id']).reset_index(drop=True)
sub18 = pd.read_csv('./out/svr_v4.csv').sort_values(['text_id']).reset_index(drop=True)


ws = {'cohesion':[-0.13600252128123766, 0.13750803056855807, -0.25347531570886483, 0.017241279751620674, 
                  0.041651234688315907, 0.3487569389867395, -0.033735200560496996, 0.07085938032892826, 
                  -0.1043149078925983, 0.24638863501517588, -0.06323499715590891, 0.3020012692487348, 
                  -0.020337771541387082, 0.17319968674174963, 0.07808033971110314, 0.0188359911733176, 
                  0.16683173516506702, 0.009746192761183298],
      
     'syntax':[-0.15031709344811364, 0.01900792042145827, -0.08401805267092738, 0.14508020053887055-0.06547, 
               0.03501598169147229, 0.25632670202629665, 0.022063607047153082, 0.05383339329056197, 
               -0.008987841890757992, 0.1335749374079727+0.06547, -0.03417666533128047, 0.22336297024869495, 
               -0.1085440896601592, 0.3112224450611823, 0.018808355058362838, -0.0689960562109341, 
               0.18393528113065485, 0.052808005289492435],
      
     'vocabulary':[-0.14895016744749542, 0.011870088811482138+0.005, -0.015798282520872665, 0.06627979580071824, 
                   0.06983348706398027, 0.3347269199249109, 0.41432258122325116, 0.08167432511176227, 
                   -0.05366977917328975, 0.13038730381536995-0.005, -0.22334263888140232, 0.11457169798848894, 
                   -0.25012169870086887, 0.20825115968594865, 0.0803685598346467, -0.06786993238015536, 
                   0.1457579883246365, 0.10170859151888886],
      
     'phraseology':[-0.21799164438187282, 0.18667199995229053, 0.04340599900575072, 0.1674699843779397, 
                    -0.1616514379571738, 0.14753759284789092+0.02564, 0.06431289063753544, 0.04801757626705726, 
                    0.07596295861157212, 0.20348939808089825, -0.15935488579667556, 0.13439692325239688, 
                    -0.0879774938946359, 0.2551375893597225-0.02564, -0.011527105688724329, -0.004508417918222821, 
                    0.16822918501783946, 0.14837888822641157],
      
     'grammar':[-0.3210656724628973, 0.11960313621035339, 0.08893375327946715, 0.036930698615279305, 
                -0.04945151334961011, 0.14725184823559215+0.05864, 0.09429257390328745, 0.09154506887489028, 
                0.009694380470424282, 0.051742560896084124, 0.05551793010912338, 0.332972773169364, 
                -0.21703293439367197, 0.22192974283335934-0.05864, 0.10930473147507612, -0.11355164593584409, 
                0.2541882037293783, 0.0871943643403443],
      
     'conventions':[-0.20105797210913556, 0.1240777234992281+0.023544, 0.20699689661095114, 0.14274473021458273, 
                    -0.04394194858775343, 0.12025664532865446, 0.06635069139382099-0.023544, 0.1407167250380728, 
                    -0.03942510555720628, 0.21949717531063806, -0.21371627554062297, 0.21399631041342992, 
                    -0.15038643745657287, 0.11413548966151718, 0.05633998341384394, -0.10482772859458124,
                    0.36053658263942917, -0.01229348567829623],
     }

columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

for col in columns:
    tmp = (rd.random() - 0.5)*0.000005
    sub1[col] = sub1[col]*ws[col][0] + sub2[col]*ws[col][1] + sub3[col]*ws[col][2] + sub4[col]*ws[col][3] + sub5[col]*ws[col][4]+\
            sub6[col]*ws[col][5] + sub7[col]*ws[col][6] + sub8[col]*ws[col][7] + sub9[col]*ws[col][8] + sub10[col]*ws[col][9]+\
            sub11[col]*ws[col][10] + sub12[col]*ws[col][11] + sub13[col]*ws[col][12] + sub14[col]*ws[col][13] + sub15[col]*ws[col][14]+\
            sub16[col]*ws[col][15] + sub17[col]*ws[col][16] + sub18[col]*ws[col][17] + tmp
sub1.to_csv('submission.csv', index=False)