## Mixer: `tropical`

**Verdict:** recall-limited (MQAR degrades as K grows)

Environment / versions:

| key | value |
| --- | --- |
| gen_version | 0.1.0 |
| python | 3.11.15 |
| platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| torch | 2.5.1+cpu |
| numpy | 2.1.3 |
| seed | 0 |

### MQAR accuracy-vs-K

| K | accuracy |
| --- | --- |
| 1 | 1.000 |
| 2 | 0.999 |
| 4 | 1.000 |
| 8 | 1.000 |
| 16 | 0.990 |
| 32 | 0.151 |

### Capability profile

| task | accuracy | steps | recurrent step |
| --- | --- | --- | --- |
| induction | 0.997 | 900 | no |
| mqar_K1 | 1.000 | 300 | no |
| mqar_K2 | 0.999 | 600 | no |
| mqar_K4 | 1.000 | 600 | no |
| mqar_K8 | 1.000 | 900 | no |
| mqar_K16 | 0.990 | 2400 | no |
| mqar_K32 | 0.151 | 3000 | no |
| selective_copy | 0.990 | 900 | no |
| parity | 0.642 | 3000 | no |
| mod_add | 1.000 | 600 | no |
| dyck | 0.547 | 3000 | no |

