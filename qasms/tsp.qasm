OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-pi) q[0];
sx q[0];
rz(2.1715000898102055) q[0];
sx q[0];
sx q[1];
rz(0.9661014017835194) q[1];
sx q[1];
cx q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.7629486307791606) q[0];
sx q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[2];
rz(-1.8408503889662615) q[2];
sx q[2];
cx q[1],q[2];
sx q[1];
rz(1.5283772515152698) q[1];
sx q[1];
cx q[0],q[1];
sx q[0];
rz(1.8962602740339056) q[0];
sx q[0];
rz(-pi) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
sx q[3];
rz(-0.4005148691131488) q[3];
sx q[3];
cx q[2],q[3];
sx q[2];
rz(0.48248750538517626) q[2];
sx q[2];
cx q[1],q[2];
sx q[1];
rz(-1.225994099264705) q[1];
sx q[1];
cx q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(2.4701374718309053) q[0];
sx q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
sx q[4];
rz(1.852121056893676) q[4];
sx q[4];
cx q[3],q[4];
sx q[3];
rz(0.2796282808700332) q[3];
sx q[3];
cx q[2],q[3];
sx q[2];
rz(2.107200760992404) q[2];
sx q[2];
cx q[1],q[2];
sx q[1];
rz(-1.0007362711536985) q[1];
sx q[1];
cx q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.8918010207396794) q[0];
sx q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[5];
rz(-2.10503952938555) q[5];
sx q[5];
cx q[4],q[5];
sx q[4];
rz(0.351968631330569) q[4];
sx q[4];
cx q[3],q[4];
sx q[3];
rz(2.7171881831496716) q[3];
sx q[3];
cx q[2],q[3];
sx q[2];
rz(-2.4412585014978525) q[2];
sx q[2];
cx q[1],q[2];
sx q[1];
rz(-0.9939498011056322) q[1];
sx q[1];
cx q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.49673755066726066) q[0];
sx q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
sx q[6];
rz(2.984284428060212) q[6];
sx q[6];
cx q[5],q[6];
sx q[5];
rz(1.1391260148421658) q[5];
sx q[5];
cx q[4],q[5];
sx q[4];
rz(1.9987758224971302) q[4];
sx q[4];
cx q[3],q[4];
sx q[3];
rz(-0.17761051582220055) q[3];
sx q[3];
cx q[2],q[3];
sx q[2];
rz(2.676981820749053) q[2];
sx q[2];
cx q[1],q[2];
rz(-pi) q[1];
sx q[1];
rz(0.2874016440911409) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
sx q[7];
rz(-2.5346814119266323) q[7];
sx q[7];
cx q[6],q[7];
sx q[6];
rz(-1.9176596128132504) q[6];
sx q[6];
cx q[5],q[6];
sx q[5];
rz(0.9599432554602529) q[5];
sx q[5];
cx q[4],q[5];
sx q[4];
rz(1.573745031244841) q[4];
sx q[4];
cx q[3],q[4];
sx q[3];
rz(-1.4558357020949595) q[3];
sx q[3];
cx q[2],q[3];
rz(-pi) q[2];
sx q[2];
rz(1.5120581888160585) q[2];
sx q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[8];
rz(0.7645100948890491) q[8];
sx q[8];
cx q[7],q[8];
sx q[7];
rz(0.5721301145179147) q[7];
sx q[7];
cx q[6],q[7];
sx q[6];
rz(-0.38189992102000936) q[6];
sx q[6];
cx q[5],q[6];
sx q[5];
rz(-1.9207605245760115) q[5];
sx q[5];
cx q[4],q[5];
sx q[4];
rz(-0.1138426920984994) q[4];
sx q[4];
cx q[3],q[4];
sx q[3];
rz(1.5200098290722215) q[3];
sx q[3];
rz(-pi) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi) q[8];
sx q[8];
rz(1.0569671805460485) q[8];
sx q[8];
cx q[7],q[8];
sx q[7];
rz(-2.006329637864594) q[7];
sx q[7];
cx q[6],q[7];
sx q[6];
rz(-0.5964656762035663) q[6];
sx q[6];
cx q[5],q[6];
sx q[5];
rz(-2.702038644878092) q[5];
sx q[5];
cx q[4],q[5];
sx q[4];
rz(1.4226183105237453) q[4];
sx q[4];
rz(-pi) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[8];
rz(0.12737742286241804) q[8];
sx q[8];
rz(-pi) q[8];
cx q[7],q[8];
sx q[7];
rz(0.8353096000629305) q[7];
sx q[7];
cx q[6],q[7];
sx q[6];
rz(1.4411261329304752) q[6];
sx q[6];
cx q[5],q[6];
sx q[5];
rz(0.4850298763754086) q[5];
sx q[5];
rz(-pi) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(-pi) q[8];
sx q[8];
rz(3.0780339189023778) q[8];
sx q[8];
cx q[7],q[8];
sx q[7];
rz(2.020159780037642) q[7];
sx q[7];
cx q[6],q[7];
rz(-pi) q[6];
sx q[6];
rz(1.8404503282605367) q[6];
sx q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
sx q[8];
rz(3.066016039153217) q[8];
sx q[8];
rz(-pi) q[8];
cx q[7],q[8];
rz(-pi) q[7];
sx q[7];
rz(1.3605086848947971) q[7];
sx q[7];
sx q[8];
rz(1.623011822388177) q[8];
sx q[8];

