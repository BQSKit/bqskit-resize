OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
sx q[0];
sx q[1];
sx q[2];
sx q[3];
sx q[4];
rz(1.7047578793081384) q[0];
rz(6.413573650172092) q[1];
rz(3.981329279609019) q[2];
rz(-1.5783117544539882) q[3];
rz(3.132211935225631) q[4];
sx q[0];
sx q[1];
sx q[2];
sx q[3];
sx q[4];
cx q[1], q[2];
rz(1.570796321823269) q[1];
sx q[2];
sx q[1];
sx q[2];
rz(0.2071275367023508) q[1];
rz(1.5707963267949043) q[2];
sx q[1];
rz(4.7123889753144566) q[1];
cx q[0], q[1];
sx q[0];
rz(7.85398163398823) q[1];
sx q[0];
sx q[1];
cx q[0], q[4];
rz(3.3487201903151695) q[1];
sx q[0];
sx q[1];
rz(1.570796332360015) q[4];
sx q[0];
sx q[4];
cx q[0], q[3];
rz(4.855651021704299) q[4];
sx q[0];
sx q[3];
sx q[4];
rz(4.5540316338991715) q[0];
sx q[3];
rz(4.712388985993064) q[4];
sx q[0];
cx q[1], q[3];
sx q[1];
sx q[3];
rz(1.5707963267126346) q[1];
sx q[3];
sx q[1];
rz(1.016139199731991) q[3];
cx q[0], q[1];
rz(-1.174648094429754) q[0];
sx q[1];
sx q[0];
rz(1.5707963268311997) q[1];
rz(1.5707963259187745) q[0];
sx q[1];
sx q[0];
rz(4.1209712455311145) q[1];
rz(4.7123889805340164) q[0];
cx q[1], q[4];
sx q[1];
rz(4.712388980384687) q[4];
sx q[1];
sx q[4];
rz(7.2625638981559675) q[1];
rz(1.427534292497631) q[4];
cx q[0], q[1];
sx q[4];
sx q[0];
sx q[1];
rz(1.5707963267948664) q[4];
rz(2.815262013232298) q[0];
sx q[1];
cx q[3], q[4];
sx q[0];
sx q[3];
rz(24.151886312164425) q[4];
cx q[0], q[1];
sx q[3];
sx q[4];
rz(4.712388980186764) q[0];
rz(7.853981632135768) q[1];
rz(1.016139219103056) q[3];
rz(2.302039256587287) q[4];
sx q[0];
sx q[1];
cx q[2], q[3];
sx q[4];
rz(1.5707963259487425) q[0];
rz(1.5707963261408286) q[1];
rz(4.712388987937011) q[2];
sx q[3];
rz(-11.782225364417885) q[4];
sx q[0];
sx q[1];
sx q[2];
rz(5.2206305838420555) q[3];
rz(2.448816105338679) q[0];
rz(1.4953758946463807) q[1];
rz(4.646222557307877) q[2];
sx q[3];
sx q[2];
cx q[2], q[3];
rz(4.712388982157094) q[2];
sx q[3];
sx q[2];
rz(3.141592653674385) q[3];
rz(5.219405404187426) q[2];
sx q[3];
sx q[2];
rz(4.712388981760531) q[2];
cx q[0], q[2];
sx q[0];
rz(4.712388980651346) q[2];
rz(3.141592653589793) q[0];
sx q[2];
sx q[0];
rz(5.776168902827868) q[2];
cx q[0], q[4];
sx q[2];
rz(1.8674246426617007) q[0];
rz(4.712388980651521) q[2];
rz(-1.5707963267949039) q[4];
sx q[0];
cx q[1], q[2];
sx q[4];
rz(5.118535219874393) q[0];
sx q[1];
rz(1.5707963267969833) q[2];
rz(5.616313803738492) q[4];
sx q[0];
sx q[1];
sx q[2];
sx q[4];
rz(0.854443152301478) q[0];
rz(4.636968542205777) q[1];
rz(1.5707963267964418) q[2];
rz(4.712388980384682) q[4];
cx q[1], q[4];
sx q[2];
sx q[1];
sx q[4];
rz(5.990347064852605) q[1];
sx q[4];
sx q[1];
cx q[3], q[4];
rz(2.2871494933124947) q[1];
rz(-4.733230785057228) q[3];
sx q[4];
sx q[3];
sx q[4];
rz(-1.305675810485305) q[3];
rz(3.141592653589793) q[4];
cx q[2], q[4];
sx q[3];
rz(1.570796326794554) q[2];
rz(-1.6501817680952557) q[3];
rz(2.9528539902235815) q[4];
sx q[2];
sx q[4];
rz(3.1664003425701432) q[2];
rz(1.9579510057708431) q[4];
sx q[2];
sx q[4];
cx q[1], q[2];
rz(3.213585654581541) q[4];
sx q[1];
sx q[2];
rz(-3.141592653413091) q[1];
sx q[2];
sx q[1];
rz(3.9960358074361317) q[1];
cx q[0], q[1];
sx q[0];
sx q[1];
rz(3.141592653589777) q[0];
rz(3.141592653589797) q[1];
sx q[0];
sx q[1];
cx q[0], q[3];
sx q[0];
rz(4.7123889803868) q[3];
rz(3.1415926535897585) q[0];
sx q[3];
sx q[0];
rz(1.3048770846748132) q[3];
rz(2.2871495012840244) q[0];
sx q[3];
cx q[0], q[4];
rz(-1.5606220471750016) q[3];
sx q[0];
sx q[4];
rz(6.990207436949209) q[0];
sx q[4];
sx q[0];
cx q[3], q[4];
rz(3.1415926535971415) q[0];
rz(6.273011027516937) q[3];
rz(0.03930028896971281) q[4];
sx q[3];
sx q[4];
rz(3.141592653590949) q[3];
rz(4.931525409020532) q[4];
sx q[3];
sx q[4];
cx q[1], q[3];
rz(6.462129031949904) q[4];
sx q[1];
sx q[3];
rz(4.8898555549010405) q[1];
sx q[3];
sx q[1];
rz(3.141592653590103) q[3];
cx q[2], q[3];
sx q[2];
sx q[3];
rz(1.3076812300515397) q[2];
rz(5.7665448818819085) q[3];
sx q[2];
sx q[3];
rz(3.141592653590274) q[3];