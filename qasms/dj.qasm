// Benchmark was created by MQT Bench on 2023-06-29
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: v1.0.0
// TKET version: 1.16.0

OPENQASM 2.0;
include "qelib1.inc";

qreg q[10];
creg c[9];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
x q[9];
x q[0];
x q[1];
x q[3];
x q[5];
x q[6];
x q[8];
h q[9];
cx q[0],q[9];
x q[0];
cx q[1],q[9];
h q[0];
x q[1];
cx q[2],q[9];
h q[1];
h q[2];
cx q[3],q[9];
x q[3];
cx q[4],q[9];
h q[3];
h q[4];
cx q[5],q[9];
x q[5];
cx q[6],q[9];
h q[5];
x q[6];
cx q[7],q[9];
h q[6];
h q[7];
cx q[8],q[9];
x q[8];
h q[8];