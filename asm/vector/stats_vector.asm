; =============================================================
; stats_vector.asm
; Version VECTORIZADA (AVX2, 8 floats por iteracion) de los
; kernels de computo. Misma ABI que la version escalar.
;
; Antes de compilar/ejecutar en su maquina, confirme soporte AVX2:
;   lscpu | grep avx2
;   cat /proc/cpuinfo | grep avx2
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n -> retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO. Fijense especialmente en:
;   (1) como se calcula cuantos elementos entran en bucles de 8
;       ("and ecx, ~7" redondea n hacia abajo al multiplo de 8),
;   (2) la REDUCCION HORIZONTAL para pasar de 8 sumas parciales
;       (un YMM) a un unico escalar,
;   (3) el BUCLE ESCALAR DE CIERRE para el remanente (n % 8 != 0).
; Reutilicen este mismo patron en compute_stats y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax               ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0       ; ymm0 = acumulador vectorial (8 carriles) = 0

    mov     ecx, esi
    and     ecx, ~7                ; ecx = n redondeado hacia abajo, multiplo de 8
    test    ecx, ecx
    jle     .sum_reduce

.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce
    vmovups ymm1, [rdi + rax*4]    ; carga 8 floats (unaligned: siempre valido)
    vaddps  ymm0, ymm0, ymm1       ; acumula por carril
    add     eax, 8
    jmp     .sum_vec_loop

.sum_reduce:
    ; --- reduccion horizontal: 8 carriles de ymm0 -> un escalar ---
    vextractf128 xmm2, ymm0, 1     ; xmm2 = mitad alta (carriles 4-7)
    vaddps  xmm0, xmm0, xmm2       ; xmm0 = 4 sumas parciales (carriles 0-3 + 4-7)
    vhaddps xmm0, xmm0, xmm0       ; suma horizontal dentro de 128 bits
    vhaddps xmm0, xmm0, xmm0       ; xmm0[0] = suma total de los 8 carriles originales

.sum_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    cmp     eax, esi
    jge     .sum_done
    vmovss  xmm1, [rdi + rax*4]
    vaddss  xmm0, xmm0, xmm1
    inc     eax
    jmp     .sum_scalar_tail

.sum_done:
    vzeroupper                     ; evita penalizacion de transicion AVX/SSE
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
; TODO (estudiante):
;   1) mean = suma(arr) / n (puede llamar a sum_array; recuerde
;      guardar arr/n/mean*/var*/min*/max* en registros callee-saved
;      antes, porque la llamada destruye registros caller-saved).
;   2) Segunda pasada VECTORIZADA para acumular sum((x-mean)^2):
;        - "broadcast" de mean a los 8 carriles con vbroadcastss.
;        - vsubps + vmulps (o vfmadd231ps si quieren ir mas alla)
;          para acumular los cuadrados de las diferencias,
;        - misma reduccion horizontal que en sum_array,
;        - bucle escalar para el remanente (subss/mulss/addss).
;   3) Min/max VECTORIZADOS con vminps/vmaxps a lo largo del bucle
;      principal, reduccion final con vextractf128 + vminps/vmaxps
;      (y shuffles si quieren reducir los 4 restantes a 1), mas
;      bucle escalar de cierre con minss/maxss o comiss.
;   4) Guarde los resultados en [rdx]=mean, [rcx]=var, [r8]=min,
;      [r9]=max. Si n == 0, escriba 0.0 en los cuatro.
;   5) 'vzeroupper' antes de cualquier 'ret' en una funcion que usa
;      registros YMM.
; ---------------------------------------------------------------

compute_stats:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    ; Guarda los argumentos en registros callee-saved porque se va a
    ; llamar a sum_array, y esa llamada destruye los registros
    ; caller-saved (rdi, rsi, rdx, rcx, r8, r9, rax, etc.)
    mov     rbx, rdi        ; rbx = arr
    mov     r12d, esi       ; r12d = n
    mov     r13, rdx        ; r13 = mean*
    mov     r14, rcx        ; r14 = var*
    mov     r15, r8         ; r15 = min*
    mov     rbp, r9         ; rbp = max*

    ;  Caso borde: n == 0 da como resultado 0.0 en los 4 punteros 
    test    r12d, r12d
    jne     .cs_calc_mean
    vxorps  xmm0, xmm0, xmm0
    vmovss  [r13], xmm0
    vmovss  [r14], xmm0
    vmovss  [r15], xmm0
    vmovss  [rbp], xmm0
    jmp     .cs_done

.cs_calc_mean:
    ;  Pasada 1: media = sum_array(arr, n) / n 
    mov     rdi, rbx
    mov     esi, r12d
    sub     rsp, 8          ; alinea la pila a 16 bytes antes del call
    call    sum_array       ; resultado (la suma) queda en xmm0
    add     rsp, 8          ; deshace el ajuste de alineacion

    vcvtsi2ss xmm1, xmm1, r12d   ; xmm1 = (float) n
    vdivss  xmm0, xmm0, xmm1     ; xmm0 = media = suma / n
    vmovss  [r13], xmm0          ; guarda la media en *mean

    ;  Pasada 2: varianza, minimo y maximo  
    mov     ecx, r12d
    and     ecx, ~7                 ; ecx = n redondeado hacia abajo, multiplo de 8
    xor     eax, eax                ; eax = i = 0

    vbroadcastss ymm1, xmm0         ; ymm1 = media repetida en los 8 carriles
    vxorps  ymm2, ymm2, ymm2        ; ymm2 = acumulador de sum((x-mean)^2) = 0

    mov     edx, 0x7F7FFFFF         ; patron de bits de +FLT_MAX
    vmovd   xmm3, edx
    vbroadcastss ymm3, xmm3         ; ymm3 = acumulador de minimos, inicia en +FLT_MAX

    mov     edx, 0xFF7FFFFF         ; patron de bits de -FLT_MAX
    vmovd   xmm4, edx
    vbroadcastss ymm4, xmm4         ; ymm4 = acumulador de maximos, inicia en -FLT_MAX

    test    ecx, ecx
    jle     .cs_reduce

.cs_vec_loop:
    cmp     eax, ecx
    jge     .cs_reduce
    vmovups ymm5, [rbx + rax*4]     ; carga 8 floats
    vsubps  ymm6, ymm5, ymm1        ; ymm6 = x - mean
    vmulps  ymm6, ymm6, ymm6        ; ymm6 = (x-mean)^2
    vaddps  ymm2, ymm2, ymm6        ; acumula suma de cuadrados
    vminps  ymm3, ymm3, ymm5        ; actualiza minimos por carril
    vmaxps  ymm4, ymm4, ymm5        ; actualiza maximos por carril
    add     eax, 8
    jmp     .cs_vec_loop

.cs_reduce:
    ; reduccion horizontal de la suma de cuadrados (igual que sum_array) 
    vextractf128 xmm7, ymm2, 1
    vaddps  xmm2, xmm2, xmm7
    vhaddps xmm2, xmm2, xmm2
    vhaddps xmm2, xmm2, xmm2        ; xmm2[0] = suma total de cuadrados

    ;  reduccion horizontal del minimo (8 carriles -> 1) 
    vextractf128 xmm7, ymm3, 1
    vminps  xmm3, xmm3, xmm7        ; 4 minimos parciales
    vshufps xmm7, xmm3, xmm3, 0xEE
    vminps  xmm3, xmm3, xmm7        ; 2 minimos parciales
    vshufps xmm7, xmm3, xmm3, 0x55
    vminps  xmm3, xmm3, xmm7        ; xmm3[0] = minimo final

    ; reduccion horizontal del maximo (mismo patron, con vmaxps) 
    vextractf128 xmm7, ymm4, 1
    vmaxps  xmm4, xmm4, xmm7
    vshufps xmm7, xmm4, xmm4, 0xEE
    vmaxps  xmm4, xmm4, xmm7
    vshufps xmm7, xmm4, xmm4, 0x55
    vmaxps  xmm4, xmm4, xmm7        ; xmm4[0] = maximo final

.cs_scalar_tail:
    ; elementos sobrantes (n % 8), van uno a la vez 
    cmp     eax, r12d
    jge     .cs_finish
    vmovss  xmm5, [rbx + rax*4]     ; x = arr[i]
    vsubss  xmm6, xmm5, xmm0        ; x - mean
    vmulss  xmm6, xmm6, xmm6        ; (x-mean)^2
    vaddss  xmm2, xmm2, xmm6
    vminss  xmm3, xmm3, xmm5
    vmaxss  xmm4, xmm4, xmm5
    inc     eax
    jmp     .cs_scalar_tail

.cs_finish:
    vcvtsi2ss xmm1, xmm1, r12d      ; xmm1 = (float) n
    vdivss  xmm2, xmm2, xmm1        ; xmm2 = varianza = suma_cuadrados / n

    vmovss  [r14], xmm2             ; *var = varianza
    vmovss  [r15], xmm3             ; *min = minimo
    vmovss  [rbp], xmm4             ; *max = maximo 

.cs_done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    vzeroupper
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual.
;
; TODO (estudiante):
;   - "Broadcast" mean y stddev a registros YMM con vbroadcastss
;     (guarde antes xmm0/xmm1 en otros registros o en la pila, ya
;     que planea usar xmm0/xmm1 tambien como temporales del bucle).
;   - Bucle vectorial de 8 en 8: vmovups/vmovaps carga, vsubps,
;     vdivps (o vmulps por el reciproco de stddev si quieren
;     optimizar), vmovups/vmovaps guarda.
;   - Bucle escalar de cierre para el remanente (n % 8), igual que
;     en sum_array.
;   - 'vzeroupper' antes del 'ret'.
; ---------------------------------------------------------------

normalize_array:
    vbroadcastss ymm2, xmm0         ; ymm2 = mean repetida en los 8 carriles
    vbroadcastss ymm3, xmm1         ; ymm3 = stddev repetida en los 8 carriles

    vxorps  xmm4, xmm4, xmm4
    vucomiss xmm1, xmm4             ; compara stddev contra 0.0
    je      .na_copy                ; si stddev == 0.0 -> copiar tal cual

    mov     ecx, edx
    and     ecx, ~7                 ; ecx = n redondeado hacia abajo, multiplo de 8
    xor     eax, eax
    test    ecx, ecx
    jle     .na_scalar_tail

.na_vec_loop:
    cmp     eax, ecx
    jge     .na_scalar_tail
    vmovups ymm5, [rdi + rax*4]     ; carga 8 floats de entrada
    vsubps  ymm5, ymm5, ymm2        ; x - mean
    vdivps  ymm5, ymm5, ymm3        ; (x - mean) / stddev
    vmovups [rsi + rax*4], ymm5     ; guarda 8 floats de salida
    add     eax, 8
    jmp     .na_vec_loop

.na_scalar_tail:
    ;  elementos sobrantes (n % 8), uno por uno
    cmp     eax, edx
    jge     .na_done
    vmovss  xmm5, [rdi + rax*4]
    vsubss  xmm5, xmm5, xmm0
    vdivss  xmm5, xmm5, xmm1
    vmovss  [rsi + rax*4], xmm5
    inc     eax
    jmp     .na_scalar_tail

.na_done:
    vzeroupper
    ret

.na_copy:
    ; caso borde: stddev == 0.0, copiar in[i] a out[i] 
    xor     eax, eax
    mov     ecx, edx
    and     ecx, ~7
    test    ecx, ecx
    jle     .na_copy_tail

.na_copy_loop:
    cmp     eax, ecx
    jge     .na_copy_tail
    vmovups ymm6, [rdi + rax*4]
    vmovups [rsi + rax*4], ymm6
    add     eax, 8
    jmp     .na_copy_loop

.na_copy_tail:
    cmp     eax, edx
    jge     .na_done
    vmovss  xmm6, [rdi + rax*4]
    vmovss  [rsi + rax*4], xmm6
    inc     eax
    jmp     .na_copy_tail
    
