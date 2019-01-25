#include "typedefs.h"

typedef struct {
  // regs
  u8 A; u8 X; u8 Y; u8 SP;
  u16 PC;
  union {
    struct { u8 C:1; u8 Z:1; u8 I:1; u8 D:1; u8 B:1; u8 u:1; u8 V:1; u8 S:1;};
    u8 P; // flags
  };
  u8  mem[0x10000]; // 64kB of mem space
  u64 cyc;
  u8  op;
  u8  op_mode;
  u8  op_bytes;
  u16 operand;

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


extern void reset(u16,u8);
extern void cpu_step(u32);
extern u8 r8(u16);
extern u8 limit_speed;
extern u8 show_debug;
