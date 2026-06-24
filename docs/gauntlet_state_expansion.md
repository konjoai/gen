## Mixer: `state_expansion`

**Verdict:** recall-limited (MQAR degrades as K grows)

Environment / versions:

| key | value |
| --- | --- |
| gen_version | 0.2.0 |
| python | 3.11.15 |
| platform | Linux-6.18.5-x86_64-with-glibc2.39 |
| torch | 2.5.1+cpu |
| numpy | 2.1.3 |
| seed | 0 |

### MQAR accuracy-vs-K

| K | accuracy |
| --- | --- |
| 1 | 1.000 |
| 2 | 0.998 |
| 4 | 0.997 |
| 8 | 1.000 |
| 16 | 1.000 |
| 32 | 0.151 |

### Capability profile

| task | accuracy | steps | recurrent step |
| --- | --- | --- | --- |
| induction | 1.000 | 600 | yes |
| mqar_K1 | 1.000 | 300 | yes |
| mqar_K2 | 0.998 | 300 | yes |
| mqar_K4 | 0.997 | 300 | yes |
| mqar_K8 | 1.000 | 600 | yes |
| mqar_K16 | 1.000 | 2100 | yes |
| mqar_K32 | 0.151 | 3000 | yes |
| selective_copy | 0.990 | 900 | yes |
| parity | 1.000 | 300 | yes |
| mod_add | 1.000 | 600 | yes |
| dyck | 1.000 | 600 | yes |

