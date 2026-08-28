```text
├── Makefile
├── include/
│   └── stats.h              # Firmas compartidas por ambas versiones
├── src/
│   └── driver.c             # Programa principal (E/S, timing, impresión)
├── asm/
│   ├── scalar/
│   │   └── stats_scalar.asm # Versión escalar (SSE escalar)
│   └── vector/
│       └── stats_vector.asm # Versión vectorial (AVX2)
├── tools/
│   ├── gen_input.py         # Genera archivos de entrada de prueba
│   └── verify_reference.py  # Verifica resultados contra referencia en Python puro
└── data/                    # Se crea al compilar: entradas/salidas .dat
