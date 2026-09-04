; =============================================================
; stats_scalar.asm
; Version ESCALAR (referencia) de los kernels de computo.
;
; Convencion de llamada: System V AMD64 ABI
;   enteros/punteros: rdi, rsi, rdx, rcx, r8, r9
;   flotantes:        xmm0, xmm1, xmm2, ...
;   retorno float:    xmm0
;   callee-saved:     rbx, rbp, r12-r15 (si los usa, debe preservarlos)
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n
;   retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO: estudien este patron (recorrido,
; acumulador, condicion de salida) antes de escribir compute_stats
; y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax           ; eax = i = 0
    xorps   xmm0, xmm0         ; xmm0 = acumulador = 0.0

.sum_loop:
    cmp     eax, esi
    jge     .sum_done
    movss   xmm1, [rdi + rax*4]
    addss   xmm0, xmm1
    inc     eax
    jmp     .sum_loop

.sum_done:
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
;   var = varianza POBLACIONAL = sum((x - mean)^2) / n
;   Caso borde: si n == 0, escriba 0.0 en mean/var/min/max.
;
; TODO (estudiante):
;   1) Calcular mean = suma(arr) / n. Puede reutilizar sum_array con
;      'call sum_array', pero recuerde que eso destruye los
;      registros caller-saved (rax, rcx, rdx, rsi, rdi, r8-r11):
;      guarde arr/n/mean*/var*/min*/max* en registros callee-saved
;      (rbx, r12-r15) ANTES de llamar.
;   2) Recorrer el arreglo una segunda vez para acumular
;      sum((x - mean)^2) y obtener var = esa suma / n.
;   3) Recorrer el arreglo (puede combinarlo con el paso 1) llevando
;      min y max con comiss + saltos condicionales (ja/jb, etc.)
;      o con las instrucciones minss/maxss.
;   4) Guardar los resultados en las direcciones recibidas por
;      puntero: [rdx]=mean, [rcx]=var, [r8]=min, [r9]=max.
;   5) No olvide restaurar los registros callee-saved en el epilogo.
; ---------------------------------------------------------------
;
;
; void compute_stats(const float *arr, int n, float *mean, float *var, float *min, float *max)
; rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*

compute_stats:
    push    rbx
    push    r12
    push    r13
    push    r14
    push    r15

    ; Caso borde: si n <= 0, escribir 0.0 en todas las salidas y terminar
    cmp     esi, 0
    jle     .handle_zero

    ; 1) Guardar argumentos en registros callee-saved ANTES de llamar a sum_array
    mov     r12, rdi       ; r12 = puntero a arr
    mov     r13d, esi      ; r13d = n
    mov     r14, rdx       ; r14 = puntero a mean
    mov     r15, rcx       ; r15 = puntero a var
    mov     rbx, r8        ; rbx = puntero a min
    push    r9             ; Guardamos r9 (puntero a max) en la pila 

    ; Llamar a sum_array (rdi y esi ya contienen arr y n correctamente)
    call    sum_array      ; Retorna la suma total en xmm0
    pop     r9             ; Restauramos el puntero a max en r9

    ; Calcular mean = suma(arr) / n
    cvtsi2ss xmm1, r13d    ; Convertimos el entero 'n' a float en xmm1
    divss   xmm0, xmm1     ; xmm0 = sum / n (media)
    movss   [r14], xmm0    ; Guardamos la media en la direccion apuntada por rdx (ahora r14)

    ; 2 y 3) Recorrer el arreglo para acumular varianza, min y max
    xorps   xmm2, xmm2     ; xmm2 = acumulador de varianza = 0.0
    movss   xmm3, [r12]    ; xmm3 = valor minimo (inicializado con arr[0])
    movss   xmm4, [r12]    ; xmm4 = valor maximo (inicializado con arr[0])

    xor     eax, eax       ; eax = indice i = 0

.stats_loop:
    cmp     eax, r13d      ; ¿i == n?
    jge     .stats_done

    movss   xmm5, [r12 + rax*4] ; xmm5 = arr[i]

    ; Actualizar minimo y maximo con instrucciones SSE dedicadas
    minss   xmm3, xmm5     ; xmm3 = min(xmm3, arr[i])
    maxss   xmm4, xmm5     ; xmm4 = max(xmm4, arr[i])

    ; Calcular (arr[i] - mean)^2 y acumular
    subss   xmm5, xmm0     ; xmm5 = arr[i] - mean
    mulss   xmm5, xmm5     ; xmm5 = (arr[i] - mean)^2
    addss   xmm2, xmm5     ; var_acc += xmm5

    inc     eax
    jmp     .stats_loop

.stats_done:
    ; 4) Guardar los resultados finales
    divss   xmm2, xmm1     ; Varianza poblacional = var_acc / n (xmm1 aun tiene 'n' en float)
    movss   [r15], xmm2    ; Guardar varianza
    movss   [rbx], xmm3    ; Guardar minimo
    movss   [r9], xmm4     ; Guardar maximo
    jmp     .epilogue

.handle_zero:
    ; Si n == 0, escribimos 0.0 en todas las direcciones
    xorps   xmm0, xmm0
    movss   [rdx], xmm0
    movss   [rcx], xmm0
    movss   [r8], xmm0
    movss   [r9], xmm0

.epilogue:
    ; 5) Restaurar registros callee-saved y retornar
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbx
    ret
; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual
;   (evite division por cero).
;
; TODO (estudiante): implementar el bucle escalar.
; Sugerencia: guarde mean (xmm0) y stddev (xmm1) en registros que no
; se sobrescriban dentro del bucle (por ejemplo xmm8/xmm9, que en
; System V no se usan para pasar argumentos), o vuelva a cargarlos
; en cada iteracion desde una copia guardada en la pila.
; ---------------------------------------------------------------
;
;
;; void normalize_array(const float *in, float *out, int n, float mean, float stddev)
; rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
normalize_array:
    ; Verificación de seguridad: si n <= 0, salir directamente
    cmp     edx, 0
    jle     .norm_end

    ; Caso borde: verificar si stddev == 0.0
    xorps   xmm2, xmm2     ; xmm2 = 0.0
    comiss  xmm1, xmm2     ; Comparamos stddev (xmm1) con 0.0 (xmm2)
    je      .stddev_zero   ; Si es cero, saltamos a la rutina especial

    ; Bucle principal de normalizacion
    xor     eax, eax       ; eax = indice i = 0

.norm_loop:
    cmp     eax, edx       ; ¿i == n?
    jge     .norm_end

    movss   xmm2, [rdi + rax*4] ; xmm2 = in[i]
    subss   xmm2, xmm0          ; xmm2 = in[i] - mean
    divss   xmm2, xmm1          ; xmm2 = (in[i] - mean) / stddev
    movss   [rsi + rax*4], xmm2 ; out[i] = xmm2

    inc     eax
    jmp     .norm_loop

.stddev_zero:
    ; Bucle alterno: si stddev == 0.0, copiar in[i] a out[i] tal cual
    xor     eax, eax       ; eax = i = 0

.zero_loop:
    cmp     eax, edx
    jge     .norm_end

    movss   xmm2, [rdi + rax*4] ; Leer in[i]
    movss   [rsi + rax*4], xmm2 ; Escribir directamente en out[i]

    inc     eax
    jmp     .zero_loop

.norm_end:
    ret
