// nv6502.cu
// A GPU implementation of a 6502 CPU emulator
// Kamil M Rocki, 1/24/19

#include "nv6502.h"
#include <stdio.h>
#include <sys/time.h>
#define CHECK_ERR_CUDA(err) if (err != cudaSuccess) { printf("%s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

int read_bin(u8* mem, const char* fname) {
  FILE * file = fopen(fname, "r+");
  if (file == NULL || mem == NULL) return - 1;
  fseek(file, 0, SEEK_END);
  long int size = ftell(file);
  fclose(file);
  file = fopen(fname, "r+");
  int bytes_read = fread(mem, sizeof(u8), size, file);
  printf("read file %s, %d bytes\n", fname, bytes_read);
  return 0; fclose(file);
}

double get_time() {
  struct timeval tv; gettimeofday(&tv, NULL);
  return (tv.tv_sec + tv.tv_usec * 1e-6);
}

__device__ __host__ void print_scrn(u32 id, _6502 *n) { for (u8 i=0;i<32;i++) { for (u8 j=0;j<32;j++) { u8 v = n->mem[0x200 + j + 32*i]; if (v >= 0xa) printf("%02x ", v); else if (v > 0) printf("%2x ", v); else printf("__ "); }  printf("\n"); } }
__device__ __host__ void print_regs(u32 id, _6502 *n) { printf("[%05d] PC: %04x OP: %02x, m: %2d, d: %04x, A:%02X X:%02X Y:%02X P:%02X SP:%02X CYC:%3ld\n", id, PC, I, m, d, A, X, Y, P, SP, CY); }

#define STACK_PG 0x0100
#define ZN(x) { Z=((x)==0); S=((x)>>7) & 0x1; }
#define LDM { d=(m>2) ? r8(n,d) : d; }
#define LD_A_OR_M() u8 w=(m==1)?A:r8(n,d)
#define ST_A_OR_M() if (m!=1) w8(n,d,w); else A=w;

__device__ u8   r8    (_6502 *n, u16 a)       { return n->mem[a % MEM_SIZE];    } // byte read
__device__ void w8    (_6502 *n, u16 a, u8 v) { n->mem[a % MEM_SIZE] = v;       } // byte write
__device__ u8   f8    (_6502 *n)              { return r8(n, PC++);  } // byte fetch
//// 16-bit versions
__device__ u16  r16   (_6502 *n, u16 a)       { u16 base=a & 0xff00; return (r8(n,a) | (r8(n,base|((u8)(a+1))) << 8)); } // buggy
__device__ u16  r16_ok(_6502 *n, u16 a)       { return (r8(n,a) | (r8(n,a+1) << 8)); }
__device__ u16  f16   (_6502 *n)              { return (f8(n) | ((f8(n))<<8)); }
//
//// stack ops
__device__ u8  pop8   (_6502 *n)              { SP++; return r8(n, STACK_PG | SP);   }
__device__ u16 pop16  (_6502 *n)              { return (pop8(n) | ((pop8(n))<<8)); }
__device__ void push8 (_6502 *n, u8 v)        { w8(n, STACK_PG | SP, v); SP--; }
__device__ void push16(_6502 *n, u16 v)       { push8(n,(v>>8)); push8(n,v);  }
__device__ void jr    (_6502 *n, u8 cond)     { if (cond) { PC=(u16)d; } }
//
//// decoding addressing mode
__device__ void imp (_6502 *n) { m=0;  b=0; } // implied, 1
__device__ void acc (_6502 *n) { m=1;  b=0; } // accumulator, 1
__device__ void imm (_6502 *n) { m=2;  b=1; d=(u16)f8(n); } // immediate, 2
__device__ void zp  (_6502 *n) { m=3;  b=1; d=(u16)f8(n); } // zero page, 2
__device__ void zpx (_6502 *n) { m=4;  b=1; u8 r=f8(n); d=(r+X) & 0xff;} // zero page, x, 3
__device__ void zpy (_6502 *n) { m=5;  b=1; u8 r=f8(n); d=(r+Y) & 0xff; } // zero page, y, 3
__device__ void rel (_6502 *n) { m=6;  b=1; u8 r=f8(n); if (r<0x80) d=PC+r; else d=PC+r-0x100;} // relative, 2
__device__ void abso(_6502 *n) { m=7;  b=2; d=f16(n); } // absolute, 3
__device__ void absx(_6502 *n) { m=8;  b=2; d=f16(n); d+=X;   } // absolute, x, 3
__device__ void absy(_6502 *n) { m=9;  b=2; d=f16(n); d+=Y;  } // absolute, y, 3
__device__ void ind (_6502 *n) { m=10; b=2; d=r16(n,f16(n)); } // indirect, 3
__device__ void indx(_6502 *n) { m=11; b=1; u8 r=f8(n); d=r16(n,(u8)(r + X)); } // indirect x
__device__ void indy(_6502 *n) { m=12; b=1; u8 r=f8(n); d=r16(n,(u8)(r)); d+=Y;} // indirect y

//instructions
__device__ void _adc(_6502 *n) {
  u8 a = A; LDM; A=d+A+C; ZN(A);
  u16 t = (u16)d + (u16)a + (u16)C; C=(t > 0xff);
  V = (!((a^d) & 0x80)) && (((a^A) & 0x80)>0 );
} //   Add Memory to Accumulator with Carry

__device__ void _sbc(_6502 *n) {
  u8 a = A; LDM; A=A-d-(1-C); ZN(A);
  s16 t = (s16)a - (s16)d - (1-(s16)C); C=(t >= 0x0);
  V = (((a^d) & 0x80)>0) && (((a^A) & 0x80)>0);
} //   Subtract Memory from Accumulator with Borrow

__device__ void _cp (_6502 *n, u8 _a, u8 _b) { u8 r=_a-_b; C=(_a>=_b); ZN(r); }
__device__ void _ora(_6502 *n) { LDM; A|=d; ZN(A); } //   "OR" Memory with Accumulator
__device__ void _and(_6502 *n) { LDM; A&=d; ZN(A); } //   "AND" Memory with Accumulator
__device__ void _eor(_6502 *n) { LDM; A^=d; ZN(A); } //   "XOR" Memory with Accumulator
__device__ void _cmp(_6502 *n) { LDM; _cp(n,A,d); } //   Compare Memory and Accumulator
__device__ void _cpx(_6502 *n) { LDM; _cp(n,X,d); } //   Compare Memory and Index X
__device__ void _cpy(_6502 *n) { LDM; _cp(n,Y,d); } //   Compare Memory and Index Y
__device__ void _bcc(_6502 *n) { jr(n,!C); } //   Branch on Carry Clear
__device__ void _bcs(_6502 *n) { jr(n,C);  } //   Branch on Carry Set
__device__ void _beq(_6502 *n) { jr(n,Z);  } //   Branch on Result Zero
__device__ void _bit(_6502 *n) { LDM; S=(d>>7) & 1; V=(d>>6) & 1; Z=(d & A)==0; } // Test Bits in Memory with A
__device__ void _bmi(_6502 *n) { jr(n, S);  } //  Branch on Result Minus
__device__ void _bne(_6502 *n) { jr(n,!Z); } //   Branch on Result not Zero
__device__ void _bpl(_6502 *n) { jr(n,!S); } //   Branch on Result Plus
__device__ void _brk(_6502 *n) { B=1;    } //   Force Break
__device__ void _bvc(_6502 *n) { jr(n,!V); } //   Branch on Overflow Clear
__device__ void _bvs(_6502 *n) { jr(n, V);  } //   Branch on Overflow Set
__device__ void _clc(_6502 *n) { C=0; } //   Clear Carry Flag
__device__ void _cld(_6502 *n) { D=0; } //   Clear Decimal Mode
__device__ void _cli(_6502 *n) { I=0; } //   Clear interrupt Disable Bit
__device__ void _clv(_6502 *n) { V=0; } //   Clear Overflow Flag
__device__ void _dec(_6502 *n) { u16 d0 = d; LDM; d--; d &= 0xff; ZN(d); w8(n,d0,d); } //   Decrement Memory by One
__device__ void _dex(_6502 *n) { X--; ZN(X); } //   Decrement Index X by One
__device__ void _dey(_6502 *n) { Y--; ZN(Y); } //   Decrement Index Y by One
__device__ void _inc(_6502 *n) { u16 d0=d; LDM; d++; d &= 0xff; ZN(d); w8(n,d0,d); d=d0; } // Incr Memory by One
__device__ void _inx(_6502 *n) { X++; ZN(X); } //   Increment Index X by One
__device__ void _iny(_6502 *n) { Y++; ZN(Y); } //   Increment Index Y by One
__device__ void _jmp(_6502 *n) { PC=d;} //   Jump to New Location
__device__ void _jsr(_6502 *n) { push16(n,PC-1); PC=d; } //   Jump to New Location Saving Return Address
__device__ void _lda(_6502 *n) { LDM; A=d; ZN(A); } //   Load Accumulator with Memory
__device__ void _ldx(_6502 *n) { LDM; X=d; ZN(X); } //   Load Index X with Memory
__device__ void _ldy(_6502 *n) { LDM; Y=d; ZN(Y); } //   Load Index Y with Memory
__device__ void _lsr(_6502 *n) { LD_A_OR_M(); C=w & 1; w>>=1; ZN(w); ST_A_OR_M(); } // Shift Right One Bit
__device__ void _asl(_6502 *n) { LD_A_OR_M(); C=(w>>7) & 1; w<<=1; ZN(w); ST_A_OR_M();} // Shift Left One Bit
__device__ void _rol(_6502 *n) { LD_A_OR_M(); u8 c = C; C=(w>>7) & 1; w=(w<<1) | c; ZN(w); ST_A_OR_M(); } // Rotate One Bit Left (Memory or Accumulator)
__device__ void _ror(_6502 *n) { LD_A_OR_M(); u8 c = C; C=(w & 1); w=(w>>1) | (c<<7); ZN(w); ST_A_OR_M(); } //   Rotate One Bit Right (Memory or Accumulator)
__device__ void _nop(_6502 *n) { /* No Operation */ }
__device__ void _pha(_6502 *n) { push8(n, A); } //   Push Accumulator on Stack
__device__ void _php(_6502 *n) { push8(n, P | 0x10); } //   Push Processor Status on Stack
__device__ void _pla(_6502 *n) { A=pop8(n); Z=(A==0); S=(A>>7)&0x1;} //   Pull Accumulator from Stack
__device__ void _plp(_6502 *n) { P=pop8(n) & 0xef | 0x20;  } //   Pull Processor Status from Stack
__device__ void _rti(_6502 *n) { P=(pop8(n) & 0xef) | 0x20; PC=pop16(n); } //   Return from Interrupt
__device__ void _rts(_6502 *n) { PC=pop16(n)+1;} //   Return from Subroutine
__device__ void _sec(_6502 *n) { C=1;} //   Set Carry Flag
__device__ void _sed(_6502 *n) { D=1;} //   Set Decimal Mode
__device__ void _sei(_6502 *n) { I=1;} //   Set Interrupt Disable Status
__device__ void _sta(_6502 *n) { w8(n,d,A);} //   Store Accumulator in Memory
__device__ void _stx(_6502 *n) { w8(n,d,X);} //   Store Index X in Memory
__device__ void _sty(_6502 *n) { w8(n,d,Y);} //   Store Index Y in Memory
__device__ void _tax(_6502 *n) { X=A; ZN(X); } //   Transfer Accumulator to Index X
__device__ void _tay(_6502 *n) { Y=A; ZN(Y); } //   Transfer Accumulator to Index Y
__device__ void _tsx(_6502 *n) { X=SP;ZN(X); } //   Transfer Stack Pointer to Index X
__device__ void _txa(_6502 *n) { A=X; ZN(A); } //   Transfer Index X to Accumulator
__device__ void _txs(_6502 *n) { SP=X; } //   Transfer Index X to Stack Pointer
__device__ void _tya(_6502 *n) { A=Y; ZN(A); } //   Transfer Index Y to Accumulator
// undocumented
__device__ void _lax(_6502 *n) { _lda(n); X=A; ZN(A); } // lda, ldx
__device__ void _sax(_6502 *n) { w8(n,d,A&X); }
__device__ void _dcp(_6502 *n) { _dec(n); _cp(n,A,d); }
__device__ void _isb(_6502 *n) { _inc(n); _sbc(n); }
__device__ void _slo(_6502 *n) { _asl(n); _ora(n); }
__device__ void _rla(_6502 *n) { _rol(n); _and(n); }
__device__ void _sre(_6502 *n) { _lsr(n); _eor(n); }
__device__ void _rra(_6502 *n) { _ror(n); _adc(n); }

__device__ void *addrtable[256] = {
  &imp, &indx,&imp,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &acc,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpx,&zpx,&imp,&absy,&imp,&absy,&absx,&absx,&absx,&absx,
  &abso,&indx,&imp,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &acc,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpx,&zpx,&imp,&absy,&imp,&absy,&absx,&absx,&absx,&absx,
  &imp, &indx,&imp,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &acc,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpx,&zpx,&imp,&absy,&imp,&absy,&absx,&absx,&absx,&absx,
  &imp, &indx,&imp,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &acc,&imm, &ind, &abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpx,&zpx,&imp,&absy,&imp,&absy,&absx,&absx,&absx,&absx,
  &imm, &indx,&imm,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &imp,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpy,&zpy,&imp,&absy,&imp,&absy,&absx,&absx,&absy,&absy,
  &imm, &indx,&imm,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &imp,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpy,&zpy,&imp,&absy,&imp,&absy,&absx,&absx,&absy,&absy,
  &imm, &indx,&imm,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &imp,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpx,&zpx,&imp,&absy,&imp,&absy,&absx,&absx,&absx,&absx,
  &imm, &indx,&imm,&indx,&zp, &zp, &zp, &zp, &imp,&imm, &imp,&imm, &abso,&abso,&abso,&abso,
  &rel, &indy,&imp,&indy,&zpx,&zpx,&zpx,&zpx,&imp,&absy,&imp,&absy,&absx,&absx,&absx,&absx};

__device__ void *optable[256] = { // opcode -> functions map
  &_brk,&_ora,&_nop,&_slo,&_nop,&_ora,&_asl,&_slo,&_php,&_ora,&_asl,&_nop,&_nop,&_ora,&_asl,&_slo,
  &_bpl,&_ora,&_nop,&_slo,&_nop,&_ora,&_asl,&_slo,&_clc,&_ora,&_nop,&_slo,&_nop,&_ora,&_asl,&_slo,
  &_jsr,&_and,&_nop,&_rla,&_bit,&_and,&_rol,&_rla,&_plp,&_and,&_rol,&_nop,&_bit,&_and,&_rol,&_rla,
  &_bmi,&_and,&_nop,&_rla,&_nop,&_and,&_rol,&_rla,&_sec,&_and,&_nop,&_rla,&_nop,&_and,&_rol,&_rla,
  &_rti,&_eor,&_nop,&_sre,&_nop,&_eor,&_lsr,&_sre,&_pha,&_eor,&_lsr,&_nop,&_jmp,&_eor,&_lsr,&_sre,
  &_bvc,&_eor,&_nop,&_sre,&_nop,&_eor,&_lsr,&_sre,&_cli,&_eor,&_nop,&_sre,&_nop,&_eor,&_lsr,&_sre,
  &_rts,&_adc,&_nop,&_rra,&_nop,&_adc,&_ror,&_rra,&_pla,&_adc,&_ror,&_nop,&_jmp,&_adc,&_ror,&_rra,
  &_bvs,&_adc,&_nop,&_rra,&_nop,&_adc,&_ror,&_rra,&_sei,&_adc,&_nop,&_rra,&_nop,&_adc,&_ror,&_rra,
  &_nop,&_sta,&_nop,&_sax,&_sty,&_sta,&_stx,&_sax,&_dey,&_nop,&_txa,&_nop,&_sty,&_sta,&_stx,&_sax,
  &_bcc,&_sta,&_nop,&_nop,&_sty,&_sta,&_stx,&_sax,&_tya,&_sta,&_txs,&_nop,&_nop,&_sta,&_nop,&_nop,
  &_ldy,&_lda,&_ldx,&_lax,&_ldy,&_lda,&_ldx,&_lax,&_tay,&_lda,&_tax,&_nop,&_ldy,&_lda,&_ldx,&_lax,
  &_bcs,&_lda,&_nop,&_lax,&_ldy,&_lda,&_ldx,&_lax,&_clv,&_lda,&_tsx,&_lax,&_ldy,&_lda,&_ldx,&_lax,
  &_cpy,&_cmp,&_nop,&_dcp,&_cpy,&_cmp,&_dec,&_dcp,&_iny,&_cmp,&_dex,&_nop,&_cpy,&_cmp,&_dec,&_dcp,
  &_bne,&_cmp,&_nop,&_dcp,&_nop,&_cmp,&_dec,&_dcp,&_cld,&_cmp,&_nop,&_dcp,&_nop,&_cmp,&_dec,&_dcp,
  &_cpx,&_sbc,&_nop,&_isb,&_cpx,&_sbc,&_inc,&_isb,&_inx,&_sbc,&_nop,&_sbc,&_cpx,&_sbc,&_inc,&_isb,
  &_beq,&_sbc,&_nop,&_isb,&_nop,&_sbc,&_inc,&_isb,&_sed,&_sbc,&_nop,&_isb,&_nop,&_sbc,&_inc,&_isb
};

// use local mem? this is usually faster
#define LMEM

__global__ void step(_6502* states, int steps, int num_threads) {
  int i = blockDim.x * blockIdx.x + threadIdx.x; // thread idx
  if (i < num_threads) {
#ifdef LMEM // use local memory
    _6502 ln = states[i]; _6502 *n = &ln;
#else       // operate directly on global mem
    _6502 *n = &states[i];
#endif

    for (int j = 0; j < steps; ++j) {
      u8 op = f8(n); I = op;               // fetch next byte
      ((void(*)(_6502*))addrtable[op])(n); // decode addr mode
      ((void(*)(_6502*))  optable[op])(n); // execute
      CY++;                                // increment cycle count
    }

#ifdef LMEM // update from local mem
    states[i] = ln;
#endif
  }
}

void reset(_6502 *n, u16 _PC, u8 _SP) {
  PC=_PC; A=0x00; X=0x00; P=0x24; SP=_SP; CY=0; memset(n->mem, '\0', MEM_SIZE);
}

// wrapper for CUDA call
void gpu_step(_6502* states, u32 steps, u32 num_blocks, u32 threads_per_block) {
  step<<<num_blocks, threads_per_block>>>(states, steps, num_blocks * threads_per_block);
}

cudaError_t err = cudaSuccess; // for checking CUDA errors
_6502* d_regs = NULL;
_6502    *h_in_regs;
_6502    *h_out_regs;
int num_blocks, threads_per_block, iters, steps, num_threads;

int init(int blks, int threads, int _iters, int _steps) {

  num_blocks = blks; threads_per_block = threads;  iters = _iters; steps = _steps;
  num_threads = num_blocks * threads_per_block;

  // allocate _6502 registers / state
  h_in_regs   = (_6502 *) malloc(num_threads * sizeof(_6502));
  h_out_regs  = (_6502 *) malloc(num_threads * sizeof(_6502));

  printf("  main: allocating %zu device bytes\n", num_threads * sizeof(_6502));
  err = cudaMalloc((void **)&d_regs, num_threads * sizeof(_6502) ); CHECK_ERR_CUDA(err);
  return 0;
}

int teardown() {

  err = cudaFree(d_regs); CHECK_ERR_CUDA(err);
  free(h_in_regs); free(h_out_regs);
  return 0;
}

int run(char* file) {
  printf("  main: running %d blocks * %d threads (%d threads total)\n", num_blocks, threads_per_block, num_threads);
  // resetting all instances
  for (u32 i = 0; i < num_threads; i++) {
    reset(&h_in_regs[i], 0x0600, 0xfe); read_bin(&h_in_regs[i].mem[0x0600],file);
  }

  printf("  main: copying host -> device\n");
  err = cudaMemcpy(d_regs, h_in_regs, sizeof(_6502 ) * num_threads, cudaMemcpyHostToDevice);  CHECK_ERR_CUDA(err);

  for (int j = 0; j < iters; j++ ) {

    double start_time = get_time();
    cudaDeviceSynchronize();
    ///
    gpu_step(d_regs, steps, num_blocks, threads_per_block);
    ///
    cudaDeviceSynchronize();
    double walltime = get_time() - start_time;
    err = cudaGetLastError(); CHECK_ERR_CUDA(err);
    printf("  main: kernel time = %.6f s, %2.6f us/step, %5.3f MHz\n", walltime,
        1e6 * (walltime/(steps * num_threads)), ((steps * num_threads)/walltime)/1e6);

    printf("  main: copying device -> host\n");
    err = cudaMemcpy(h_out_regs, d_regs, sizeof(_6502) * num_threads, cudaMemcpyDeviceToHost); CHECK_ERR_CUDA(err);

    for (u32 i = 0; i < num_threads; i++) { print_regs(i, &h_out_regs[i]); print_scrn(i, &h_out_regs[i]); }
  }

  return 0;
}

int main(int argc, void** argv) {
  init(1,1,32,1024);
  run("sierp.bin");
  teardown();
}

