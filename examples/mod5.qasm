OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
sx q[0];
sx q[1];
sx q[2];
sx q[3];
sx q[4];
rz(1.5707963275750216) q[0];
rz(4.712388980384604) q[1];
sx q[2];
sx q[3];
rz(4.7123889803119665) q[4];
sx q[0];
sx q[1];
rz(2.356194490192351) q[3];
sx q[4];
cx q[2], q[3];
sx q[2];
rz(2.356194564260934) q[3];
rz(4.712388980384713) q[2];
sx q[3];
sx q[2];
rz(1.5707962892926202) q[3];
cx q[1], q[2];
sx q[3];
sx q[1];
sx q[2];
rz(1.5707963939907827) q[3];
cx q[0], q[3];
rz(4.712388980384679) q[1];
rz(7.853981633952762) q[2];
sx q[0];
sx q[1];
sx q[2];
sx q[3];
sx q[0];
rz(8.00461850378553) q[1];
rz(0.36792810985495594) q[2];
rz(2.356194490166851) q[3];
sx q[3];
cx q[2], q[3];
sx q[2];
sx q[3];
sx q[2];
rz(5.497787143827556) q[3];
sx q[3];
cx q[0], q[3];
sx q[0];
sx q[3];
sx q[0];
sx q[3];
cx q[0], q[2];
rz(-1.5707963267964096) q[3];
sx q[0];
sx q[2];
sx q[0];
sx q[2];
rz(3.926990816940283) q[2];
cx q[0], q[2];
sx q[0];
rz(1.1533262732794514) q[2];
sx q[0];
sx q[2];
rz(0.2574528635395989) q[0];
rz(1.5707963273603447) q[2];
cx q[0], q[1];
sx q[2];
sx q[0];
rz(7.068583470573475) q[1];
cx q[2], q[4];
sx q[0];
sx q[1];
sx q[2];
rz(1.5707963267948963) q[4];
rz(4.173564122473413) q[0];
rz(1.5707963267984344) q[1];
sx q[2];
sx q[4];
sx q[1];
rz(9.424777960752616) q[2];
rz(0.78539816338523) q[4];
cx q[1], q[2];
sx q[4];
sx q[1];
rz(4.712388978855006) q[2];
rz(1.5707963267948961) q[4];
rz(1.5707963267948832) q[1];
sx q[2];
cx q[3], q[4];
sx q[1];
rz(0.03630382461765346) q[2];
sx q[3];
sx q[4];
rz(2.3561944901922907) q[1];
sx q[2];
sx q[3];
rz(-1.5707963265745326) q[4];
cx q[0], q[1];
rz(4.712388978855009) q[2];
sx q[4];
sx q[0];
rz(5.497787143782071) q[1];
sx q[0];
sx q[1];
rz(4.712388980384682) q[1];
sx q[1];
cx q[1], q[2];
sx q[1];
sx q[2];
rz(1.5707963267948464) q[1];
sx q[2];
sx q[1];
rz(1.5707963362595747) q[2];
cx q[0], q[2];
rz(3.7763539471761964) q[1];
sx q[0];
sx q[2];
sx q[0];
rz(0.7853981586437012) q[2];
sx q[2];
cx q[0], q[2];
sx q[0];
sx q[2];
rz(3.141592653588824) q[0];
rz(-0.749094334352559) q[2];
sx q[0];
sx q[2];
rz(-0.03692162474867125) q[0];
rz(1.5707963357862338) q[2];
cx q[2], q[3];
sx q[2];
sx q[3];
rz(4.71238898038469) q[2];
rz(-1.5707963267948968) q[3];
sx q[2];
sx q[3];
cx q[2], q[4];
rz(3.141592653597535) q[3];
sx q[2];
rz(0.7853981633973667) q[4];
sx q[2];
sx q[4];
rz(4.872649062015501) q[2];
rz(1.5707963268208738) q[4];
sx q[4];
rz(4.712388980409454) q[4];
cx q[0], q[4];
sx q[0];
sx q[4];
sx q[0];
rz(3.926990816984922) q[4];
sx q[4];
cx q[2], q[4];
sx q[2];
sx q[4];
sx q[2];
rz(2.356194490194663) q[4];
sx q[4];
cx q[0], q[4];
sx q[0];
rz(4.712388980385268) q[4];
sx q[0];
sx q[4];
cx q[0], q[2];
rz(1.5707963267948923) q[4];
sx q[0];
sx q[2];
sx q[4];
sx q[0];
sx q[2];
rz(3.9269908163549383) q[2];
cx q[0], q[2];
rz(4.66458779745142) q[0];
sx q[2];
sx q[0];
rz(9.42477795984489) q[2];
rz(7.853981634029369) q[0];
sx q[2];
sx q[0];
rz(0.9456582451861728) q[2];
