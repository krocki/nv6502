// nv6502.h
// Kamil M Rocki, 1/24/19

#include "typedefs.h"

#define MEM_SIZE 0x1000 // size of private memory per cpu

// a struct holding the complete state of one 6502 core
typedef struct {
  // regs
  u8 A; u8 X; u8 Y; u8 SP;
  u16 PC; // program counter
  union {
    struct { u8 C:1; u8 Z:1; u8 I:1; u8 D:1; u8 B:1; u8 u:1; u8 V:1; u8 S:1;};
    u8 P; // flags
  };
  u8  mem[MEM_SIZE];// private mem space
  u64 cyc;          // cycle counter
  u8  op;           // current opcode
  u8  op_mode;      // current addressing mode
  u8  op_bytes;     // how many arg bytes to fetch?
  u16 operand;      // arg bytes
  u64 max_cycles;
} _6502;

#define SP (n->SP)
#define PC (n->PC)
#define  A (n->A)
#define  X (n->X)
#define  Y (n->Y)
#define  P (n->P)
#define  C (n->C)
#define  Z (n->Z)
#define  I (n->I)
#define  D (n->D)
#define  B (n->B)
#define  V (n->V)
#define  S (n->S)
#define CY (n->cyc)
#define  I (n->op)
#define  m (n->op_mode)
#define  b (n->op_bytes)
#define  d (n->operand)
